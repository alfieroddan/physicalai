# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run separate control and inference graphs connected by Zenoh.

Run from the Runtime repository root on Linux:
    PYTHONPATH=src:proposal/demos uv run --no-project --with 'loguru>=0.7' --with 'eclipse-zenoh==1.10.2' python proposal/demos/two_process/zenoh_two_process_demo.py

Open robotgraph-two-process-trace.json in https://ui.perfetto.dev.
"""

import json
import subprocess
import sys
from argparse import ArgumentParser
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from queue import Full, Queue
from select import select
from signal import SIGINT
from tempfile import TemporaryDirectory

import zenoh
from _robotgraph_trace_common import AsyncPolicy, control_graph
from _robotgraph_zenoh import ZenohWorker, declare_service, peer_config, serve_query

from physicalai.graph import Graph, GraphExecutor, Message, Node, NodeContext, compile_graph

TRACE_PATH = Path("robotgraph-two-process-trace.json")


class _InferenceNode(Node):
    def __init__(self, endpoint: str) -> None:
        super().__init__("Serve inference request", rate=100)
        self.endpoint = endpoint

    def start(self, _ctx: NodeContext, /) -> None:
        self.requests: Queue[zenoh.Query] = Queue(maxsize=1)
        with ExitStack() as stack:
            server = stack.enter_context(zenoh.open(peer_config(self.endpoint, listen=True)))
            stack.enter_context(declare_service(server, self._enqueue))
            self.server = stack.pop_all()
        print("ready", flush=True)

    def _enqueue(self, query: zenoh.Query) -> None:
        # The Zenoh callback hands off the request; only the graph runs the model.
        try:
            self.requests.put_nowait(query)
        except Full:
            query.reply_err(b"Inference queue is full")

    def ready(self, _ctx: NodeContext, /) -> bool:
        return not self.requests.empty()

    def step(self, ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        serve_query(ctx, self.requests.get_nowait())
        return {}

    def stop(self, _ctx: NodeContext, /) -> None:
        self.server.close()


def _run_inference(endpoint: str, trace_path: str) -> None:
    graph = Graph()
    graph.add(_InferenceNode(endpoint))
    try:
        GraphExecutor(compile_graph(graph)).run(duration_s=60, logging=True, trace_path=trace_path)
    except KeyboardInterrupt:
        # The launcher stops this process after the control graph has finished.
        pass


def _merge_traces(control: Path, inference: Path) -> None:
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in (control, inference)]
    origins = [trace["otherData"]["monotonic_origin_ns"] for trace in traces]
    origin = min(origins)
    events = []
    for trace, start, name in zip(traces, origins, ("Control graph", "Inference graph"), strict=True):
        for event in trace["traceEvents"]:
            if "ts" in event:
                # Both processes run on the same host and share its monotonic clock.
                event["ts"] += (start - origin) / 1_000  # Trace timestamps are in microseconds.
            if event["ph"] == "M" and event["name"] == "process_name":
                event["args"]["name"] = name
            events.append(event)
    dropped = sum(trace["otherData"]["dropped_events"] for trace in traces)
    merged = {"traceEvents": events, "otherData": {"dropped_events": dropped}}
    TRACE_PATH.write_text(json.dumps(merged), encoding="utf-8")


def _stop_child(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        child.send_signal(SIGINT)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        raise


def _run_control() -> None:
    with TemporaryDirectory(prefix="robotgraph-two-process-") as directory:
        workspace = Path(directory)
        endpoint = f"unixsock-stream/{workspace / 'peer.sock'}"
        control_trace = workspace / "control.json"
        inference_trace = workspace / "inference.json"
        command = [sys.executable, __file__, "--inference", endpoint, str(inference_trace)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True) as child:
            try:
                pipe = child.stdout
                if pipe is None or not select([pipe], [], [], 5)[0] or pipe.readline().strip() != "ready":
                    msg = "Inference graph did not become ready"
                    raise RuntimeError(msg)
                policy = AsyncPolicy(ZenohWorker(endpoint), local_warmup=False)
                graph = control_graph(policy)
                GraphExecutor(compile_graph(graph)).run(duration_s=2, logging=True, trace_path=control_trace)
            finally:
                _stop_child(child)
            if child.returncode != 0:
                msg = f"Inference graph exited with status {child.returncode}"
                raise RuntimeError(msg)
        _merge_traces(control_trace, inference_trace)
    print(f"Open {TRACE_PATH} in https://ui.perfetto.dev")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inference", nargs=2, metavar=("ENDPOINT", "TRACE_PATH"))
    args = parser.parse_args()
    if args.inference:
        _run_inference(*args.inference)
    else:
        _run_control()
