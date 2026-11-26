# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run separate control and inference graphs connected by Zenoh.

Run from the Runtime repository root on Linux:
    PYTHONPATH=src:proposal/demos uv run --no-project --with 'loguru>=0.7' --with 'eclipse-zenoh==1.10.2' python proposal/demos/two_process/zenoh_two_process_demo.py

Open robotgraph-two-process-trace.json in https://ui.perfetto.dev.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from queue import Full, Queue

import zenoh
from _launch import run_pair
from _robotgraph_trace_common import AsyncPolicy, control_graph
from _robotgraph_zenoh import ZenohWorker, declare_service, peer_config, serve_query

from physicalai.graph import Graph, Message, Node, NodeContext

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


def _inference_graph(endpoint: str) -> Graph:
    graph = Graph()
    graph.add(_InferenceNode(endpoint))
    return graph


def _control_graph(endpoint: str) -> Graph:
    return control_graph(AsyncPolicy(ZenohWorker(endpoint), local_warmup=False))


if __name__ == "__main__":
    run_pair(__file__, _control_graph, _inference_graph, trace_path=TRACE_PATH)
