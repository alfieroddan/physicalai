# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a compiled graph on the calling thread."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from math import isfinite
from threading import TIMEOUT_MAX, Event, Lock
from time import monotonic_ns
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from physicalai.graph.compiler import NANOSECONDS_PER_SECOND
from physicalai.graph.model import Captured, Message, NodeContext, Timestamp

from ._trace import record_trace

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from physicalai.graph.compiler import ExecutionPlan, NodePlan
    from physicalai.graph.model import Node, PortAddress

    from ._trace import Trace


class Clock(Protocol):
    """The clock and interruptible wait used by the executor."""

    def now(self) -> Timestamp:
        """Return the current monotonic time."""

    def wait(self, stop: Event, timeout_s: float) -> None:
        """Wait for the timeout or a stop request, whichever comes first."""


class _MonotonicClock:
    @staticmethod
    def now() -> Timestamp:
        return Timestamp(monotonic_ns(), "monotonic")

    @staticmethod
    def wait(stop: Event, timeout_s: float) -> None:
        stop.wait(min(timeout_s, TIMEOUT_MAX))


class NodeError(RuntimeError):
    """A node hook or its output validation failed."""


@contextmanager
def _node_hook(node: Node, hook: str) -> Generator[None, None, None]:
    try:
        yield
    except Exception as error:
        msg = f"Node {node.name!r} failed in {hook}: {error}"
        raise NodeError(msg) from error


class GraphExecutor:
    """Run one session at a time, with fresh channels on each run."""

    def __init__(self, plan: ExecutionPlan, *, clock: Clock | None = None) -> None:
        """Set the compiled plan and an optional clock override."""
        self._plan = plan
        self._clock = _MonotonicClock() if clock is None else clock
        self._lock = Lock()
        self._stop: Event | None = None

    def stop(self) -> None:
        """Request that the active run stop after its current hook returns."""
        with self._lock:
            if self._stop is not None:
                self._stop.set()

    def run(
        self, *, duration_s: float, logging: bool = False, trace_path: str | Path = "robotgraph-trace.json"
    ) -> None:
        """Run for up to duration_s after startup, or until stopped or failed.

        A failed node hook or invalid output ends the run with a NodeError.
        With logging enabled, write a metadata-only Perfetto trace to trace_path.

        Raises:
            ValueError: The duration is negative or outside the supported range.
            RuntimeError: This executor is already running.
        """
        duration_ns = duration_s * NANOSECONDS_PER_SECOND
        if not isfinite(duration_ns) or duration_ns < 0:
            msg = "Expected a finite, non-negative duration in seconds"
            raise ValueError(msg)

        with self._lock:
            if self._stop is not None:
                msg = "Graph executor already running"
                raise RuntimeError(msg)
            stop = self._stop = Event()

        try:
            tracing = record_trace(trace_path, self._clock.now) if logging else nullcontext()
            with tracing as trace:
                _Session(self._plan, self._clock, stop, trace).run(round(duration_ns))
        finally:
            with self._lock:
                self._stop = None


class _Session:
    def __init__(self, plan: ExecutionPlan, clock: Clock, stop: Event, trace: Trace | None = None) -> None:
        self.plan = plan
        self.clock = clock
        self.stop = stop
        self.trace = trace
        trace_span = trace.span if trace is not None else None
        self.contexts = {entry.node.name: NodeContext(entry.node.name, clock.now, trace_span) for entry in plan.nodes}
        # An absent destination is an empty slot. Each destination has just one producer.
        self.latest: dict[PortAddress, Message[object]] = {}
        self.sequences: dict[PortAddress, int] = {}

    def run(self, duration_ns: int) -> None:
        started: list[Node] = []
        failure: BaseException | None = None
        try:
            self._start(started)
            self._schedule(duration_ns)
        except BaseException as error:
            failure = error
            raise
        finally:
            self._cleanup(started, failure)

    def _start(self, started: list[Node]) -> None:
        for entry in self.plan.nodes:
            if self.stop.is_set():
                break
            node = entry.node
            with _node_hook(node, "start"):
                node.start(self.contexts[node.name])
            started.append(node)

    def _cleanup(self, started: list[Node], failure: BaseException | None) -> None:
        first_error = failure
        for node in reversed(started):
            try:
                with _node_hook(node, "stop"):
                    node.stop(self.contexts[node.name])
            except BaseException as error:  # ruff: ignore[BLE001] - Attempt all stops despite failures.
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(f"Cleanup failed for {node.name!r}: {error}")
        if failure is None and first_error is not None:
            raise first_error

    def _schedule(self, duration_ns: int) -> None:
        if not self.plan.nodes:
            return
        start = self.clock.now().ns
        end = start + duration_ns
        due_times = dict.fromkeys((entry.node.name for entry in self.plan.nodes), start)

        while not self.stop.is_set():
            now = self.clock.now().ns
            if now >= end:
                return
            next_tick = min(due_times.values())
            if next_tick > now:
                self.clock.wait(self.stop, (min(next_tick, end) - now) / NANOSECONDS_PER_SECOND)
                continue

            # Use one time snapshot: work that becomes due mid-batch waits for the next pass.
            for entry in self.plan.nodes:
                name = entry.node.name
                scheduled = due_times[name]
                if scheduled > now:
                    continue
                started = self.clock.now().ns
                if self.stop.is_set() or started >= end:
                    return
                finished = self._tick(entry, end, scheduled, started)
                if finished > scheduled + entry.period_ns:
                    logger.warning("Graph node {} overran its period", name)

                # Advance on the original cadence, never enqueue missed ticks.
                periods = (finished - scheduled) // entry.period_ns + 1
                due_times[name] = scheduled + periods * entry.period_ns

    def _tick(self, entry: NodePlan, end: int, scheduled: int, started: int) -> int:
        if self.trace is None:
            self._step(entry, end)
            return self.clock.now().ns

        outcome = "failed"
        input_ages_ns: dict[str, int | None] = {}
        try:
            outcome = self._step(entry, end, input_ages_ns)
        finally:
            finished = self.clock.now().ns
            self.trace.tick(
                entry.node.name,
                scheduled_ns=scheduled,
                started_ns=started,
                finished_ns=finished,
                period_ns=entry.period_ns,
                outcome=outcome,
                input_ages_ns=input_ages_ns,
            )
        return finished

    def _step(self, entry: NodePlan, end: int, input_ages_ns: dict[str, int | None] | None = None) -> str:
        node = entry.node
        ctx = self.contexts[node.name]
        with _node_hook(node, "ready"):
            if not node.ready(ctx):
                return "not ready"

        inputs = {}
        now = self.clock.now()
        for edge in entry.incoming:
            message = self.latest.get(edge.destination.address)
            port = edge.destination.name
            max_age = edge.policy.max_age if edge.policy is not None else None
            age_ns = None
            if message is not None and (max_age is not None or input_ages_ns is not None):
                age_ns = self._age_ns(message, now)
            if input_ages_ns is not None:
                input_ages_ns[port] = age_ns

            fresh = max_age is None or (age_ns is not None and age_ns / NANOSECONDS_PER_SECOND <= max_age)
            if message is not None and fresh:
                inputs[port] = message
            elif edge.destination.required:
                return "missing input" if message is None else "stale input"

        if self.stop.is_set() or self.clock.now().ns >= end:
            return "stopped"
        with _node_hook(node, "step"):
            self._publish(entry, node.step(ctx, inputs))
        return "ran"

    @staticmethod
    def _age_ns(message: Message[object], now: Timestamp) -> int | None:
        timestamp = message.captured_at or message.published_at
        if now.clock_domain != timestamp.clock_domain or timestamp.ns > now.ns:
            return None
        return now.ns - timestamp.ns

    def _publish(self, entry: NodePlan, outputs: Mapping[str, object]) -> None:
        if not isinstance(outputs, Mapping):
            msg = "Expected a mapping of output port names to payloads"
            raise TypeError(msg)

        # Validate all outputs before changing a channel or sequence number.
        messages: dict[PortAddress, Message[object]] = {}
        published_at = self.clock.now()
        for name, value in outputs.items():
            port = entry.node.output(name)
            payload, captured_at = value, None
            if isinstance(value, Captured):
                payload, captured_at = value.payload, value.captured_at
                if not isinstance(captured_at, Timestamp):
                    msg = f"Output {port.address} expects a Timestamp for capture time"
                    raise TypeError(msg)
            if not isinstance(payload, port.data_type):
                msg = f"Output {port.address} expects {port.data_type.__name__}, got {type(payload).__name__}"
                raise TypeError(msg)
            sequence = self.sequences.get(port.address, 0) + 1
            messages[port.address] = Message(port.address, sequence, captured_at, published_at, payload)

        for address, message in messages.items():
            self.sequences[address] = message.sequence
        for edge in entry.outgoing:
            message = messages.get(edge.source.address)
            if message is not None:
                self.latest[edge.destination.address] = message
