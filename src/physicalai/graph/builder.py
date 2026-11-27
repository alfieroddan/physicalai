# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build and compile graphs with named nodes and ports."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from physicalai.graph.compiler import compile_plan
from physicalai.graph.model import Edge, Node

if TYPE_CHECKING:
    from physicalai.graph.compiler import ExecutionPlan
    from physicalai.graph.model import Port, PortAddress

NodeT = TypeVar("NodeT", bound=Node)


class Graph:
    """A mutable graph builder; incomplete wiring is allowed until compilation."""

    def __init__(self) -> None:
        """Create an empty graph."""
        self._nodes: dict[str, Node] = {}
        self._edges: dict[PortAddress, Edge] = {}

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Nodes in registration order."""
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[Edge, ...]:
        """The registered edges."""
        return tuple(self._edges.values())

    def add(self, node: NodeT) -> NodeT:
        """Register a node for wiring.

        Returns:
            The registered node.

        Raises:
            TypeError: The argument is not a node.
            ValueError: The name is already registered.
        """
        if not isinstance(node, Node):
            msg = "Graph only accepts Node instances"
            raise TypeError(msg)
        if node.name in self._nodes:
            msg = f"Duplicate node name: {node.name!r}"
            raise ValueError(msg)
        self._nodes[node.name] = node
        return node

    def connect(self, source: Port, destination: Port) -> Edge:
        """Connect two ports with the default delivery policy.

        Returns:
            The registered edge.
        """
        return self.add_edge(Edge(source, destination))

    def add_edge(self, edge: Edge) -> Edge:
        """Register an edge between declared ports on registered nodes.

        Returns:
            The registered edge.

        Raises:
            ValueError: The endpoints are invalid or the input already has a producer.
        """
        source, destination = edge.source, edge.destination
        label = f"{source.address} -> {destination.address}"
        if source.direction != "output" or destination.direction != "input":
            msg = f"Expected output -> input: {label}"
            raise ValueError(msg)
        for port in (source, destination):
            node = self._nodes.get(port.node.name)
            if node is not port.node or not any(declared is port for declared in node.ports):
                msg = f"Unknown endpoint {port.address}: {label}"
                raise ValueError(msg)
        if destination.address in self._edges:
            msg = f"Input already has a producer: {label}"
            raise ValueError(msg)
        self._edges[destination.address] = edge
        return edge

    def compile(self) -> ExecutionPlan:
        """Validate the graph without starting nodes or opening channels.

        Returns:
            The fixed schedule and routes for the graph.
        """
        return compile_plan(self.nodes, self.edges)
