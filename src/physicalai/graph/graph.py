# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build and compile graphs with named nodes and ports."""

from pathlib import Path
from typing import TypeVar

from physicalai.graph.types import Edge, Node, Port, PortAddress

# Type variable for nodes, to preserve the specific node type.
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
        # Validate that the source and destination ports belong to registered nodes.
        source, destination = edge.source, edge.destination
        label = f"{source.address} -> {destination.address}"

        # Require that the source and destination ports are correctly oriented.
        if source.direction != "output" or destination.direction != "input":
            msg = f"Expected output -> input: {label}"
            raise ValueError(msg)

        # Sanity check that the ports belong to the nodes and are declared.
        for port in (source, destination):
            node = self._nodes.get(port.node.name)
            if node is not port.node or node is None or not any(declared is port for declared in node.ports):
                msg = f"Unknown endpoint {port.address}: {label}"
                raise ValueError(msg)

        # Check data type compatibility between the source and destination ports.
        if source.data_type is not destination.data_type:
            msg = f"Data type mismatch: {label}"
            raise ValueError(msg)

        # Ensure that the destination port does not already have an incoming edge.
        if destination.address in self._edges:
            msg = f"Input already has a producer: {label}"
            raise ValueError(msg)

        self._edges[destination.address] = edge
        return edge

    def __str__(self) -> str:
        """Summarize the registered nodes and connections.

        Returns:
            A string summarizing the graph's nodes and edges.
        """
        nodes, edges = self.nodes, self.edges
        names = ", ".join(node.name for node in nodes) or "(none)"
        lines = [
            f"Graph ({len(nodes)} nodes, {len(edges)} edges)",
            f"  Nodes: {names}",
        ]
        for edge in edges:
            source, destination = edge.source, edge.destination
            lines.append(
                f"  {source.node.name}.{source.name} -> "
                f"{destination.node.name}.{destination.name} "
                f"[{source.data_type.__name__}]"
            )
        return "\n".join(lines)

    def export_svg(self, path: str | Path) -> Path:
        """Save the current graph as an SVG and return its path.

        Returns:
            The path to the saved SVG file.

        Raises:
            RuntimeError: If Graphviz is not installed.
        """
        # raise error if graphviz not installed
        try:
            import graphviz  # noqa: PLC0415
        except ImportError as e:
            msg = "Graphviz is required to export SVG"
            raise RuntimeError(msg) from e

        diagram = graphviz.Digraph("graph")
        diagram.attr(rankdir="LR")
        diagram.attr("node", shape="box")

        node_ids = {}
        for index, node in enumerate(self.nodes):
            node_id = f"n{index}"
            node_ids[node.name] = node_id
            diagram.node(node_id, label=graphviz.escape(node.name))

        for edge in self.edges:
            source, destination = edge.source, edge.destination
            diagram.edge(
                node_ids[source.node.name],
                node_ids[destination.node.name],
                label=graphviz.escape(f"{source.name} → {destination.name}"),
            )

        destination_path = Path(path)
        svg = diagram.pipe(format="svg", encoding="utf-8")
        destination_path.write_text(svg, encoding="utf-8")
        return destination_path
