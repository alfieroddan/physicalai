# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fake devices and graph wiring shared by the sync and async demos."""

from collections.abc import Mapping
from random import Random
from time import sleep
from typing import ClassVar, cast

from physicalai.graph import Graph, GraphExecutor, Message, Node, NodeContext, compile_graph

RATE_HZ = 30
CHUNK_SIZE = 8
REQUEST_THRESHOLD = 0.5
_IO_RNG = Random(42)
_MODEL_RNG = Random(42)


def _delay_ms(mean: float, stddev: float) -> None:
    sleep(max(0.0, _IO_RNG.gauss(mean, stddev)) / 1_000)


class _FakeCamera(Node):
    outputs: ClassVar[Mapping[str, type]] = {"frame": int}

    def start(self, _ctx: NodeContext, /) -> None:
        self.frame = 0

    def step(self, _ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        _delay_ms(6, 1)
        self.frame += 1
        return {"frame": self.frame}


class _FakeRobot(Node):
    outputs: ClassVar[Mapping[str, type]] = {"position": float}

    def start(self, _ctx: NodeContext, /) -> None:
        self.position = 0.0

    def step(self, _ctx: NodeContext, _inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        _delay_ms(2, 0.5)
        return {"position": self.position}

    def send_action(self, action: float) -> None:
        _delay_ms(1, 0.2)
        self.position = action


def predict_action_chunk(frame: int, position: float) -> tuple[float, ...]:
    """Simulate a model taking about 100 ms to predict an action chunk."""
    sleep(max(0.0, _MODEL_RNG.gauss(100, 10)) / 1_000)
    return (position + frame / 1_000,) * CHUNK_SIZE


class _SendAction(Node):
    inputs: ClassVar[Mapping[str, type]] = {"action": float}

    def __init__(self, robot: _FakeRobot) -> None:
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


def run_demo(policy: Node, trace_path: str) -> None:
    """Run the same read → policy → send graph for either policy."""
    graph = Graph()
    camera = graph.add(_FakeCamera("Camera read", rate=RATE_HZ))
    robot = graph.add(_FakeRobot("Robot state read", rate=RATE_HZ))
    graph.add(policy)
    send = graph.add(_SendAction(robot))
    graph.connect(camera.output("frame"), policy.input("frame"))
    graph.connect(robot.output("position"), policy.input("position"))
    graph.connect(policy.output("action"), send.input("action"))

    GraphExecutor(compile_graph(graph)).run(duration_s=2, logging=True, trace_path=trace_path)
    print(f"Open {trace_path} in https://ui.perfetto.dev")
