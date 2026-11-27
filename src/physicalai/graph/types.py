# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Graph types and related abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

PortAddress = tuple[str, str]


@dataclass(frozen=True)
class Port:
    """A declared input or output socket of a node."""

    node: Node
    name: str
    direction: Literal["input", "output"]
    data_type: type

    @property
    def address(self) -> PortAddress:
        """The node and port names."""
        return (self.node.name, self.name)


@dataclass(frozen=True)
class Edge:
    """Connect a named output to a named input."""

    source: Port
    destination: Port


class Node:
    """A graph node."""

    inputs: ClassVar[Mapping[str, type]] = {}
    outputs: ClassVar[Mapping[str, type]] = {}

    def __init__(self, name: str) -> None:
        """Initialize the node with a name."""
        self._name = name

        self._inputs = {port: Port(self, port, "input", data_type) for port, data_type in self.inputs.items()}
        self._outputs = {port: Port(self, port, "output", data_type) for port, data_type in self.outputs.items()}

    @property
    def name(self) -> str:
        """The name of the node."""
        return self._name

    @property
    def ports(self) -> tuple[Port, ...]:
        """The declared input and output ports."""
        return (*self._inputs.values(), *self._outputs.values())

    def input(self, name: str) -> Port:
        """Look up an input port by name.

        Returns:
            The input port.
        """
        return self._inputs[name]

    def output(self, name: str) -> Port:
        """Look up an output port by name.

        Returns:
            The output port.
        """
        return self._outputs[name]
