# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fake devices, graph wiring, and async control policy for the demos."""

from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from random import Random
from time import sleep
from typing import ClassVar, Protocol, cast

from physicalai.graph import Graph, GraphExecutor, Message, Node, NodeContext

RATE_HZ = 30
CHUNK_SIZE = 8
REQUEST_THRESHOLD = 0.5
_IO_RNG = Random(42)
_MODEL_RNG = Random(42)
ActionChunk = tuple[float, ...]


def _delay_ms(mean: float, stddev: float) -> None:
    sleep(max(0.0, _IO_RNG.gauss(mean, stddev)) / 1_000)


class FakeCamera(Node):
    outputs: ClassVar[Mapping[str, type]] = {"frame": int}

    def start(self, _ctx: NodeContext, /) -> None:
        self.frame = 0

    def step(self, _ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        _delay_ms(6, 1)
        self.frame += 1
        return {"frame": self.frame}


class FakeRobot(Node):
    outputs: ClassVar[Mapping[str, type]] = {"position": float}

    def start(self, _ctx: NodeContext, /) -> None:
        self.position = 0.0

    def step(self, _ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        _delay_ms(2, 0.5)
        return {"position": self.position}

    def send_action(self, action: float) -> None:
        _delay_ms(1, 0.2)
        self.position = action


def predict_action_chunk(frame: int, position: float) -> ActionChunk:
    """Simulate a model taking about 100 ms to predict an action chunk."""
    sleep(max(0.0, _MODEL_RNG.gauss(100, 10)) / 1_000)
    return (position + frame / 1_000,) * CHUNK_SIZE


def infer(ctx: NodeContext, frame: int, position: float) -> ActionChunk:
    with ctx.trace_span("Model inference"):
        return predict_action_chunk(frame, position)


class Worker(Protocol):
    def start(self, ctx: NodeContext) -> None: ...
    def submit(self, frame: int, position: float) -> None: ...
    def poll(self) -> ActionChunk | None: ...
    def stop(self) -> None: ...


class ThreadWorker:
    def __init__(self, predict: Callable[[NodeContext, int, float], ActionChunk] = infer) -> None:
        self.predict = predict

    def start(self, ctx: NodeContext) -> None:
        self.ctx = ctx
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Inference")
        self.pending: Future[ActionChunk] | None = None

    def submit(self, frame: int, position: float) -> None:
        self.pending = self.pool.submit(self.predict, self.ctx, frame, position)

    def poll(self) -> ActionChunk | None:
        pending = self.pending
        if pending is None or not pending.done():
            return None
        self.pending = None
        return pending.result()

    def stop(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.pending is not None and not self.pending.cancelled():
            self.poll()  # Surface a worker failure even if the run has ended.


class AsyncPolicy(Node):
    inputs: ClassVar[Mapping[str, type]] = {"frame": int, "position": float}
    outputs: ClassVar[Mapping[str, type]] = {"action": float}

    def __init__(self, worker: Worker, *, local_warmup: bool = True) -> None:
        super().__init__("Policy: request or pop", rate=RATE_HZ)
        self.worker = worker
        self.local_warmup = local_warmup

    def start(self, ctx: NodeContext, /) -> None:
        self.in_flight = False
        self.actions: deque[float] = deque()
        self.last_action: float | None = None
        self.total_pops = 0
        self.pops_at_request = 0
        self.threshold = int(CHUNK_SIZE * REQUEST_THRESHOLD)
        self.worker.start(ctx)

    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        frame = cast("int", inputs["frame"].payload)
        position = cast("float", inputs["position"].payload)

        # Like Runtime, seed the queue once on the control thread.
        if self.local_warmup and self.last_action is None:
            with ctx.trace_span("Model warmup"):
                self.actions = deque(predict_action_chunk(frame, position))

        chunk = self.worker.poll() if self.in_flight else None
        if chunk is not None:
            self.in_flight = False
            # Drop predictions for ticks that consumed actions while inference ran.
            offset = min(self.total_pops - self.pops_at_request, len(chunk) - 1)
            self.actions = deque(chunk[offset:])

        if len(self.actions) < self.threshold and not self.in_flight:
            self.pops_at_request = self.total_pops
            self.worker.submit(frame, position)
            self.in_flight = True

        if self.actions:
            self.last_action = self.actions.popleft()
            self.total_pops += 1
        # If inference is late, send the last action without counting another pop.
        return {} if self.last_action is None else {"action": self.last_action}

    def stop(self, _ctx: NodeContext, /) -> None:
        self.worker.stop()


class SendAction(Node):
    inputs: ClassVar[Mapping[str, type]] = {"action": float}

    def __init__(self, robot: FakeRobot) -> None:
        super().__init__("Send to robot", rate=RATE_HZ)
        self.robot = robot

    def start(self, _ctx: NodeContext, /) -> None:
        self.last_sequence = 0

    def step(self, _ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        action = inputs["action"]
        # Latest reads may return the same message if the policy was not due.
        if action.sequence != self.last_sequence:
            self.robot.send_action(cast("float", action.payload))
            self.last_sequence = action.sequence
        return {}


def control_graph(policy: Node) -> Graph:
    """Wire the same camera and robot nodes around the selected policy."""
    graph = Graph()
    camera = graph.add(FakeCamera("Camera read", rate=RATE_HZ))
    robot = graph.add(FakeRobot("Robot state read", rate=RATE_HZ))
    graph.add(policy)
    send = graph.add(SendAction(robot))
    graph.connect(camera.output("frame"), policy.input("frame"))
    graph.connect(robot.output("position"), policy.input("position"))
    graph.connect(policy.output("action"), send.input("action"))
    return graph


def run_demo(policy: Node, trace_path: str) -> None:
    """Run the same read → policy → send graph for either policy."""
    graph = control_graph(policy)
    GraphExecutor(graph.compile()).run(duration_s=2, logging=True, trace_path=trace_path)
    print(f"Open {trace_path} in https://ui.perfetto.dev")
