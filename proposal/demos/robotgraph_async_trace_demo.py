# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trace a 30 Hz control loop with threshold-based refill on a background thread.

Run from the Runtime repository root:
    PYTHONPATH=src uv run --no-project --with 'loguru>=0.7' python proposal/demos/robotgraph_async_trace_demo.py

Open robotgraph-async-trace.json in https://ui.perfetto.dev (Open trace file).
"""

from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ClassVar, cast

from _robotgraph_trace_common import CHUNK_SIZE, RATE_HZ, REQUEST_THRESHOLD, predict_action_chunk, run_demo

from physicalai.graph import Message, Node, NodeContext


def _infer(ctx: NodeContext, frame: int, position: float) -> tuple[float, ...]:
    with ctx.trace_span("Model inference"):
        return predict_action_chunk(frame, position)


class _AsyncPolicy(Node):
    inputs: ClassVar[Mapping[str, type]] = {"frame": int, "position": float}
    outputs: ClassVar[Mapping[str, type]] = {"action": float}

    def start(self, _ctx: NodeContext, /) -> None:
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Inference")
        self.pending: Future[tuple[float, ...]] | None = None
        self.actions: deque[float] = deque()
        self.last_action: float | None = None
        self.total_pops = 0
        self.pops_at_request = 0
        self.threshold = int(CHUNK_SIZE * REQUEST_THRESHOLD)

    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        frame = cast("int", inputs["frame"].payload)
        position = cast("float", inputs["position"].payload)

        # Like Runtime, seed the queue once on the control thread.
        if self.last_action is None:
            with ctx.trace_span("Model warmup"):
                self.actions = deque(predict_action_chunk(frame, position))

        pending = self.pending
        if pending is not None and pending.done():
            self.pending = None
            chunk = pending.result()
            # Drop predictions for ticks that consumed actions while inference ran.
            offset = min(self.total_pops - self.pops_at_request, len(chunk) - 1)
            self.actions = deque(chunk[offset:])

        if len(self.actions) < self.threshold and self.pending is None:
            self.pops_at_request = self.total_pops
            self.pending = self.worker.submit(_infer, ctx, frame, position)

        if self.actions:
            self.last_action = self.actions.popleft()
            self.total_pops += 1
        # If inference is late, send the last action without counting another pop.
        return {"action": self.last_action}

    def stop(self, _ctx: NodeContext, /) -> None:
        self.worker.shutdown(wait=True, cancel_futures=True)
        if self.pending is not None and not self.pending.cancelled():
            self.pending.result()  # Surface a worker failure even if the run has ended.


if __name__ == "__main__":
    run_demo(_AsyncPolicy("Policy: request or pop", rate=RATE_HZ), "robotgraph-async-trace.json")
