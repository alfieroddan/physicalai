# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compile graph wiring into a fixed, single-worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import isfinite
from typing import TYPE_CHECKING, Any

from physicalai.graph.model import Latest

if TYPE_CHECKING:
    from physicalai.graph.model import Edge, Node

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class NodePlan:
    """The period and routes for one node."""

    node: Node
    period_ns: int
    incoming: tuple[Edge, ...]
    outgoing: tuple[Edge, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    """An immutable schedule and routing table for one worker."""

    nodes: tuple[NodePlan, ...]


def _period_ns(node: Node) -> int:
    rate = node.rate
    if rate is None or not isfinite(rate) or rate <= 0:
        msg = f"{node.name}: expected a finite, positive rate in Hz"
        raise ValueError(msg)

    period = NANOSECONDS_PER_SECOND / rate
    if not isfinite(period) or period < 1:
        msg = f"{node.name}: rate is outside the clock's supported range"
        raise ValueError(msg)
    return round(period)


def _validate_edge(edge: Edge) -> None:
    source, destination = edge.source, edge.destination
    label = f"{source.address} -> {destination.address}"
    if not issubclass(source.data_type, destination.data_type):
        msg = f"Incompatible payload types: {label}"
        raise TypeError(msg)

    if edge.policy is not None:
        if not isinstance(edge.policy, Latest):
            msg = f"Unsupported delivery policy: {label}"
            raise TypeError(msg)
        age = edge.policy.max_age
        if age is not None and (not isfinite(age) or age < 0):
            msg = f"Expected a finite, non-negative max_age: {label}"
            raise ValueError(msg)


def _validate_inputs(plan: NodePlan) -> None:
    wired = {edge.destination.name for edge in plan.incoming}
    for port in plan.node.ports:
        if not port.name or port.data_type is Any or not isinstance(port.data_type, type):
            msg = f"Expected a named port with a Python class as payload type: {port.address}"
            raise TypeError(msg)
        if port.direction == "input" and port.required and port.name not in wired:
            msg = f"Missing required input: {port.address}"
            raise ValueError(msg)


def compile_plan(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> ExecutionPlan:
    """Validate node declarations and wiring for the standalone executor.

    Returns:
        The fixed schedule and routes for the graph.

    Raises:
        ValueError: A timing value is invalid, an input is missing, or wiring has a cycle.
    """
    indices = {node.name: index for index, node in enumerate(nodes)}
    incoming: list[list[Edge]] = [[] for _ in nodes]
    outgoing: list[list[Edge]] = [[] for _ in nodes]
    for edge in edges:
        incoming[indices[edge.destination.node.name]].append(edge)
        outgoing[indices[edge.source.node.name]].append(edge)

    plans = []
    for index, node in enumerate(nodes):
        plan = NodePlan(node, _period_ns(node), tuple(incoming[index]), tuple(outgoing[index]))
        _validate_inputs(plan)
        plans.append(plan)

    for edge in edges:
        _validate_edge(edge)

    # A heap makes registration order the tie-breaker, even for newly unblocked nodes.
    remaining = [len(routes) for routes in incoming]
    ready = [index for index, count in enumerate(remaining) if count == 0]
    ordered = []
    while ready:
        index = heappop(ready)
        ordered.append(plans[index])
        for edge in outgoing[index]:
            consumer = indices[edge.destination.node.name]
            remaining[consumer] -= 1
            if remaining[consumer] == 0:
                heappush(ready, consumer)

    if len(ordered) != len(nodes):
        blocked = [node.name for index, node in enumerate(nodes) if remaining[index]]
        msg = f"Graph contains a cycle; blocked nodes: {', '.join(blocked)}"
        raise ValueError(msg)
    return ExecutionPlan(tuple(ordered))
