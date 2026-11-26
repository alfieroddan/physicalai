# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run fast S1 control and slow S2 planning in two local Zenoh-connected processes.

Run from the Runtime repository root on Linux:
    PYTHONPATH=src:proposal/demos uv run --no-project --with 'loguru>=0.7' --with 'eclipse-zenoh==1.10.2' python proposal/demos/two_process/s1s2_zenoh_demo.py

Open robotgraph-s1s2-trace.json in https://ui.perfetto.dev.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from struct import Struct
from threading import Lock
from time import sleep
from typing import ClassVar, cast

import zenoh
from _launch import run_pair
from _robotgraph_trace_common import RATE_HZ, FakeCamera, FakeRobot, SendAction
from _robotgraph_zenoh import peer_config

from physicalai.graph import Captured, Edge, Graph, Latest, Message, Node, NodeContext, Timestamp

FRAMES = "robotgraph/demo/s1s2/frames"
INTENTS = "robotgraph/demo/s1s2/intents"
PACKET = Struct("!Qq")  # Monotonic timestamp in nanoseconds, then a frame number or goal.
S2_HZ = 1
FRAME_MAX_AGE_S = 0.25
INTENT_MAX_AGE_S = 1.5


class _Receive(Node):
    outputs: ClassVar[Mapping[str, type]] = {"value": int}

    def __init__(self, name: str, endpoint: str, topic: str, *, rate: float, listen: bool) -> None:
        super().__init__(name, rate=rate)
        self.endpoint = endpoint
        self.topic = topic
        self.listen = listen

    def start(self, _ctx: NodeContext, /) -> None:
        self.lock = Lock()
        self.latest: tuple[int, int] | None = None
        with ExitStack() as stack:
            self.session = stack.enter_context(zenoh.open(peer_config(self.endpoint, listen=self.listen)))
            stack.enter_context(self.session.declare_subscriber(self.topic, self._receive))
            self.resources = stack.pop_all()
        if self.listen:
            print("ready", flush=True)

    def _receive(self, sample: zenoh.Sample) -> None:
        if len(sample.payload) == PACKET.size:
            value = PACKET.unpack(sample.payload.to_bytes())
            with self.lock:
                self.latest = value

    def step(self, ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        # Zenoh callbacks only replace the inbox. The executor publishes from it.
        with self.lock:
            sample, self.latest = self.latest, None
        if sample is None:
            return {}
        timestamp_ns, value = sample
        return {"value": Captured(value, Timestamp(timestamp_ns, ctx.now().clock_domain))}

    def stop(self, _ctx: NodeContext, /) -> None:
        self.resources.close()


class _Publish(Node):
    inputs: ClassVar[Mapping[str, type]] = {"value": int}

    def __init__(self, name: str, peer: _Receive, topic: str, *, rate: float) -> None:
        super().__init__(name, rate=rate)
        self.peer = peer
        self.topic = topic

    def start(self, _ctx: NodeContext, /) -> None:
        self.last_sequence = 0

    def step(self, _ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        message = inputs["value"]
        if message.sequence != self.last_sequence:
            timestamp = message.captured_at or message.published_at
            data = PACKET.pack(timestamp.ns, cast("int", message.payload))
            self.peer.session.put(self.topic, data, congestion_control=zenoh.CongestionControl.DROP)
            self.last_sequence = message.sequence
        return {}


class _S2(Node):
    inputs: ClassVar[Mapping[str, type]] = {"frame": int}
    outputs: ClassVar[Mapping[str, type]] = {"goal": int}

    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        frame = cast("int", inputs["frame"].payload)
        with ctx.trace_span("S2 planning"):
            sleep(0.20)
            return {"goal": -1 if (frame // RATE_HZ) % 2 else 1}


class _S1(Node):
    inputs: ClassVar[Mapping[str, type]] = {"frame": int, "position": float, "goal": int}
    outputs: ClassVar[Mapping[str, type]] = {"action": float}

    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        # A fresh frame gates the step. The toy policy moves toward an allowed goal.
        goal = cast("int", inputs["goal"].payload)
        if goal not in (-1, 1):
            return {}
        position = cast("float", inputs["position"].payload)
        with ctx.trace_span("S1 policy"):
            sleep(0.004)
            return {"action": position + 0.1 * (goal - position)}


def _control_graph(endpoint: str) -> Graph:
    graph = Graph()
    camera = graph.add(FakeCamera("Camera read", rate=RATE_HZ))
    robot = graph.add(FakeRobot("Robot state read", rate=RATE_HZ))
    goals = graph.add(_Receive("Read S2 intent", endpoint, INTENTS, rate=RATE_HZ, listen=False))
    frames = graph.add(_Publish("Publish camera frame", goals, FRAMES, rate=RATE_HZ))
    policy = graph.add(_S1("S1: compute action", rate=RATE_HZ))
    send = graph.add(SendAction(robot))

    graph.connect(camera.output("frame"), frames.input("value"))
    graph.add_edge(Edge(camera.output("frame"), policy.input("frame"), Latest(max_age=FRAME_MAX_AGE_S)))
    graph.connect(robot.output("position"), policy.input("position"))
    graph.add_edge(Edge(goals.output("value"), policy.input("goal"), Latest(max_age=INTENT_MAX_AGE_S)))
    graph.add_edge(Edge(policy.output("action"), send.input("action"), Latest(max_age=0.1)))
    return graph


def _planning_graph(endpoint: str) -> Graph:
    graph = Graph()
    frames = graph.add(_Receive("Read camera frame", endpoint, FRAMES, rate=S2_HZ, listen=True))
    planner = graph.add(_S2("S2: choose goal", rate=S2_HZ))
    goals = graph.add(_Publish("Publish S2 intent", frames, INTENTS, rate=S2_HZ))

    graph.add_edge(Edge(frames.output("value"), planner.input("frame"), Latest(max_age=FRAME_MAX_AGE_S)))
    graph.connect(planner.output("goal"), goals.input("value"))
    return graph


if __name__ == "__main__":
    run_pair(
        __file__,
        _control_graph,
        _planning_graph,
        duration_s=3.5,
        trace_path=Path("robotgraph-s1s2-trace.json"),
        labels=("S1 control", "S2 planner"),
    )
