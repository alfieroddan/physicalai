# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trace a 30 Hz control loop with synchronous, threshold-based action refill.

Run from the Runtime repository root:
    PYTHONPATH=src uv run --no-project --with 'loguru>=0.7' python proposal/demos/robotgraph_sync_trace_demo.py

Open robotgraph-sync-trace.json in https://ui.perfetto.dev (Open trace file).
"""

from collections import deque
from collections.abc import Mapping
from typing import ClassVar, cast

from _robotgraph_trace_common import CHUNK_SIZE, RATE_HZ, REQUEST_THRESHOLD, predict_action_chunk, run_demo

from physicalai.graph import Message, Node, NodeContext


class _SyncPolicy(Node):
    inputs: ClassVar[Mapping[str, type]] = {"frame": int, "position": float}
    outputs: ClassVar[Mapping[str, type]] = {"action": float}

    def start(self, _ctx: NodeContext, /) -> None:
        self.actions: deque[float] = deque()
        self.threshold = max(1, int(CHUNK_SIZE * REQUEST_THRESHOLD))

    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        # Check before popping, and replace the old tail on refill.
        if len(self.actions) < self.threshold:
            frame = cast("int", inputs["frame"].payload)
            position = cast("float", inputs["position"].payload)
            with ctx.trace_span("Model inference"):
                self.actions = deque(predict_action_chunk(frame, position))
        return {"action": self.actions.popleft()}


if __name__ == "__main__":
    run_demo(_SyncPolicy("Policy: refill or pop", rate=RATE_HZ), "robotgraph-sync-trace.json")
