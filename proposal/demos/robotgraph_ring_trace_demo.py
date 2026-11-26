# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pass observations and action chunks between threads through shared rings.

Run from the Runtime repository root:
    PYTHONPATH=src uv run --no-project --with 'loguru>=0.7' python proposal/demos/robotgraph_ring_trace_demo.py

Open robotgraph-ring-trace.json in https://ui.perfetto.dev (Open trace file).
"""

from threading import Semaphore, Thread
from typing import Generic, TypeVar

from _robotgraph_trace_common import ActionChunk, AsyncPolicy, infer, run_demo

from physicalai.graph import NodeContext

T = TypeVar("T")


class _Ring(Generic[T]):
    """Fixed slots shared by one producer and one consumer in the same process."""

    def __init__(self, capacity: int) -> None:
        self.slots: list[T | None] = [None] * capacity
        self.read_index = 0
        self.write_index = 0
        self.free = Semaphore(capacity)
        self.ready = Semaphore(0)

    def put(self, item: T) -> None:
        if not self.free.acquire(blocking=False):
            msg = "Inference ring is full"
            raise RuntimeError(msg)
        self.slots[self.write_index] = item
        self.write_index = (self.write_index + 1) % len(self.slots)
        self.ready.release()

    def get(self, *, wait: bool) -> T | None:
        if not self.ready.acquire(blocking=wait):
            return None
        item = self.slots[self.read_index]
        self.slots[self.read_index] = None
        self.read_index = (self.read_index + 1) % len(self.slots)
        self.free.release()
        return item


class _RingWorker:
    def start(self, ctx: NodeContext) -> None:
        # One request may be in flight, plus a shutdown sentinel.
        self.requests = _Ring[tuple[int, float] | None](2)
        self.replies = _Ring[ActionChunk | Exception](1)
        self.thread = Thread(target=self._serve, args=(ctx,), name="Ring inference")
        self.thread.start()

    def submit(self, frame: int, position: float) -> None:
        self.requests.put((frame, position))

    def poll(self) -> ActionChunk | None:
        reply = self.replies.get(wait=False)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _serve(self, ctx: NodeContext) -> None:
        while (observation := self.requests.get(wait=True)) is not None:
            try:
                chunk = infer(ctx, *observation)
            except Exception as error:
                self.replies.put(error)
                return
            self.replies.put(chunk)

    def stop(self) -> None:
        self.requests.put(None)
        self.thread.join()
        self.poll()  # Surface a worker failure even if the run has ended.


if __name__ == "__main__":
    run_demo(AsyncPolicy(_RingWorker()), "robotgraph-ring-trace.json")
