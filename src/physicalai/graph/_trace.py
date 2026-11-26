# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Write metadata-only Chrome trace JSON for the Perfetto UI."""

from __future__ import annotations

import json
from contextlib import contextmanager
from os import getpid
from pathlib import Path
from threading import Lock, current_thread, get_native_id
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from physicalai.graph.model import Timestamp

_MAX_EVENTS = 100_000
_NS_PER_MICROSECOND = 1_000
_NS_PER_MILLISECOND = 1_000_000


class Trace:
    """Buffer a bounded number of timing events across threads."""

    def __init__(self, now: Callable[[], Timestamp]) -> None:
        """Set a shared time origin and label the control lane."""
        self._now = now
        self._origin_ns = now().ns
        self._pid = getpid()
        self._lock = Lock()
        tid = get_native_id()
        self._threads = {tid}
        self._events: list[dict[str, object]] = [
            {"ph": "M", "name": "process_name", "pid": self._pid, "args": {"name": "RobotGraph"}},
            self._thread_name(tid, "Graph control"),
        ]
        self._dropped = 0

    def _thread_name(self, tid: int, name: str) -> dict[str, object]:
        return {"ph": "M", "name": "thread_name", "pid": self._pid, "tid": tid, "args": {"name": name}}

    def tick(
        self, name: str, *, scheduled_ns: int, started_ns: int, finished_ns: int, period_ns: int, outcome: str
    ) -> None:
        """Record one tick, including delay caused by preceding synchronous work."""
        elapsed_ns = finished_ns - scheduled_ns
        args: dict[str, object] = {
            "outcome": outcome,
            "start_delay_ms": (started_ns - scheduled_ns) / _NS_PER_MILLISECOND,
            "period_ms": period_ns / _NS_PER_MILLISECOND,
            "deadline_missed": elapsed_ns > period_ns,
            "missed_ticks": elapsed_ns // period_ns,
        }
        self._append_span(name, started_ns, finished_ns, args)

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        """Record work on the calling thread without capturing its data.

        Yields:
            Control to the timed block.
        """
        started = self._now().ns
        try:
            yield
        finally:
            self._append_span(name, started, self._now().ns, {})

    def _append_span(self, name: str, started_ns: int, finished_ns: int, args: dict[str, object]) -> None:
        tid = get_native_id()
        # Chrome trace timestamps and durations are in microseconds.
        event: dict[str, object] = {
            "ph": "X",
            "name": name,
            "cat": "robotgraph",
            "pid": self._pid,
            "tid": tid,
            "ts": (started_ns - self._origin_ns) / _NS_PER_MICROSECOND,
            "dur": (finished_ns - started_ns) / _NS_PER_MICROSECOND,
            "args": args,
        }
        with self._lock:
            new_thread = tid not in self._threads
            if len(self._events) + int(new_thread) >= _MAX_EVENTS:
                self._dropped += 1
                return
            if new_thread:
                self._events.append(self._thread_name(tid, current_thread().name))
                self._threads.add(tid)
            self._events.append(event)

    def write(self, path: str | Path) -> None:
        """Write the buffered trace after execution, without touching payloads."""
        with self._lock:
            data = {"traceEvents": self._events.copy(), "otherData": {"dropped_events": self._dropped}}
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(data, file)


@contextmanager
def record_trace(path: str | Path, now: Callable[[], Timestamp]) -> Generator[Trace, None, None]:
    """Save the trace on exit, without masking an execution failure.

    Yields:
        A trace for recording ticks during the run.
    """
    trace = Trace(now)
    try:
        yield trace
    except BaseException as error:
        try:
            trace.write(path)
        except OSError as write_error:
            error.add_note(f"Could not write graph trace: {write_error}")
        raise
    else:
        trace.write(path)
