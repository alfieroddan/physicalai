# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Launch two local graphs and align their traces on the host's monotonic clock."""

import json
import subprocess
import sys
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from select import select
from signal import SIGINT
from tempfile import TemporaryDirectory

from physicalai.graph import Graph, GraphExecutor, compile_graph


def _merge_traces(paths: tuple[Path, Path], labels: tuple[str, str], output: Path) -> None:
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    origins = [trace["otherData"]["monotonic_origin_ns"] for trace in traces]
    origin = min(origins)
    events = []
    for trace, start, name in zip(traces, origins, labels, strict=True):
        for event in trace["traceEvents"]:
            if "ts" in event:
                event["ts"] += (start - origin) / 1_000  # Trace timestamps are in microseconds.
            if event["ph"] == "M" and event["name"] == "process_name":
                event["args"]["name"] = name
            events.append(event)
    dropped = sum(trace["otherData"]["dropped_events"] for trace in traces)
    merged = {"traceEvents": events, "otherData": {"dropped_events": dropped}}
    output.write_text(json.dumps(merged), encoding="utf-8")


def _stop_child(child: subprocess.Popen[str]) -> None:
    if child.poll() is None:
        child.send_signal(SIGINT)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        raise


def run_pair(
    script: str,
    control: Callable[[str], Graph],
    inference: Callable[[str], Graph],
    *,
    trace_path: Path,
    duration_s: float = 2,
    labels: tuple[str, str] = ("Control graph", "Inference graph"),
) -> None:
    """Run the parent or child role; the child must print `ready` after binding Zenoh."""
    parser = ArgumentParser()
    parser.add_argument("--inference", nargs=2, metavar=("ENDPOINT", "TRACE_PATH"))
    args = parser.parse_args()
    if args.inference:
        endpoint, child_trace = args.inference
        try:
            GraphExecutor(compile_graph(inference(endpoint))).run(duration_s=60, logging=True, trace_path=child_trace)
        except KeyboardInterrupt:
            pass  # The parent stops the child after control finishes.
        return

    with TemporaryDirectory(prefix="robotgraph-two-process-") as directory:
        workspace = Path(directory)
        endpoint = f"unixsock-stream/{workspace / 'peer.sock'}"
        control_trace = workspace / "control.json"
        inference_trace = workspace / "inference.json"
        command = [sys.executable, script, "--inference", endpoint, str(inference_trace)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True) as child:
            try:
                pipe = child.stdout
                if pipe is None or not select([pipe], [], [], 5)[0] or pipe.readline().strip() != "ready":
                    msg = "Inference graph did not become ready"
                    raise RuntimeError(msg)
                GraphExecutor(compile_graph(control(endpoint))).run(
                    duration_s=duration_s, logging=True, trace_path=control_trace
                )
            finally:
                _stop_child(child)
            if child.returncode != 0:
                msg = f"Inference graph exited with status {child.returncode}"
                raise RuntimeError(msg)
        _merge_traces((control_trace, inference_trace), labels, trace_path)
    print(f"Open {trace_path} in https://ui.perfetto.dev")
