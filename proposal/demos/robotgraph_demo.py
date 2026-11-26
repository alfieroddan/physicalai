# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Show a 4 Hz counter feeding two 2 Hz recorders.

Run from the Runtime repository root:
    PYTHONPATH=src uv run --no-project --with 'loguru>=0.7' python proposal/demos/robotgraph_demo.py
"""

from collections.abc import Mapping
from typing import ClassVar

from physicalai.graph import Graph, GraphExecutor, Message, Node, NodeContext, compile_graph


class _Counter(Node):
    outputs: ClassVar[Mapping[str, type]] = {"count": int}

    def start(self, _ctx: NodeContext, /) -> None:
        self.count = 0

    def step(self, _ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        self.count += 1
        return {"count": self.count}


class _Recorder(Node):
    inputs: ClassVar[Mapping[str, type]] = {"sample": int}

    def step(self, _ctx: NodeContext, inputs: Mapping[str, Message[object]]) -> Mapping[str, object]:
        print(f"{self.name}: {inputs['sample'].payload}")
        return {}


if __name__ == "__main__":
    graph = Graph()
    counter = graph.add(_Counter("counter", rate=4))
    left = graph.add(_Recorder("left", rate=2))
    right = graph.add(_Recorder("right", rate=2))

    # Each recorder reads the latest count on its own tick.
    graph.connect(counter.output("count"), left.input("sample"))
    graph.connect(counter.output("count"), right.input("sample"))

    GraphExecutor(compile_graph(graph)).run(duration_s=1)
