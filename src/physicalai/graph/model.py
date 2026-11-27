# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Nodes, typed ports, edges, and messages for a local execution graph."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from contextlib import AbstractContextManager

T = TypeVar("T")
PortAddress = tuple[str, str]


@dataclass(frozen=True)
class Timestamp:
    """Time in nanoseconds within a named clock domain."""

    ns: int
    clock_domain: str


@dataclass(frozen=True)
class Captured(Generic[T]):
    """An output payload with a known capture time."""

    payload: T
    captured_at: Timestamp


@dataclass(frozen=True)
class Message(Generic[T]):
    """A published payload with source and timing metadata."""

    source: PortAddress
    sequence: int
    captured_at: Timestamp | None
    published_at: Timestamp
    payload: T


@dataclass(frozen=True)
class NodeContext:
    """The node name and the executor's clock for lifecycle hooks."""

    node_name: str
    now: Callable[[], Timestamp]
    _trace_span: Callable[[str], AbstractContextManager[None]] | None = None

    def trace_span(self, name: str) -> AbstractContextManager[None]:
        """Record named work on the calling thread when tracing is enabled.

        Returns:
            A context manager for the span. Use metadata-only names.
        """
        return nullcontext() if self._trace_span is None else self._trace_span(name)


@dataclass(frozen=True)
class Port:
    """A declared input or output endpoint."""

    node: Node
    name: str
    direction: Literal["input", "output"]
    data_type: type
    required: bool = True

    @property
    def address(self) -> PortAddress:
        """The node and port names."""
        return (self.node.name, self.name)


class Node:
    """A unit of work with named input and output types."""

    inputs: ClassVar[Mapping[str, type]] = {}
    outputs: ClassVar[Mapping[str, type]] = {}
    optional_inputs: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, name: str, *, rate: float | None = None) -> None:
        """Set the node name and rate in Hz, and snapshot its port declarations.

        Raises:
            ValueError: The name is empty or an optional input is undeclared.
        """
        if not name or not self.optional_inputs <= self.inputs.keys():
            msg = f"Invalid node name or optional input declaration: {name!r}"
            raise ValueError(msg)

        self._name = name
        self._rate = rate
        self._inputs = {
            port: Port(self, port, "input", data_type, port not in self.optional_inputs)
            for port, data_type in self.inputs.items()
        }
        self._outputs = {port: Port(self, port, "output", data_type) for port, data_type in self.outputs.items()}

    @property
    def name(self) -> str:
        """The graph-local node name."""
        return self._name

    @property
    def rate(self) -> float | None:
        """The requested periodic rate in Hz."""
        return self._rate

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

    def start(self, ctx: NodeContext, /) -> None:
        """Set up the node. If setup fails, clean up partial work before raising."""

    def ready(self, _ctx: NodeContext, /) -> bool:  # ruff: ignore[PLR6301] - The default hook is stateless.
        """Return whether the node can step without waiting for setup."""
        return True

    def step(self, ctx: NodeContext, inputs: Mapping[str, Message[object]], /) -> Mapping[str, object]:
        """Process the inputs and return new payloads keyed by output port name."""
        raise NotImplementedError

    def stop(self, ctx: NodeContext, /) -> None:
        """Release resources after a successful start."""


@dataclass(frozen=True)
class Latest:
    """Keep the most recent message, with an optional maximum age in seconds."""

    max_age: float | None = None


@dataclass(frozen=True)
class Edge:
    """Connect a named output to a named input."""

    source: Port
    destination: Port
    policy: Latest | None = None
