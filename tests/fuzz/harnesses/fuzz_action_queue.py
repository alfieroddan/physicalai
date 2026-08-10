# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz ChunkedActionQueue — concurrent push/pop must not deadlock, crash, or return non-1D arrays.
Chunks are pre-generated before threads start so FuzzedDataProvider is single-threaded.
"""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.runtime.execution.queue import ChunkedActionQueue
    from physicalai.runtime.smoothers import LerpSmoother, ReplaceSmoother


def test_one_input(data: bytes) -> None:
    if len(data) < 8:
        return

    fdp = atheris.FuzzedDataProvider(data)
    use_lerp = fdp.ConsumeBool()
    smoother = (
        LerpSmoother(duration_frames=fdp.ConsumeIntInRange(0, 16))
        if use_lerp
        else ReplaceSmoother()
    )
    queue = ChunkedActionQueue(smoother=smoother)

    action_dim = fdp.ConsumeIntInRange(0, 16)
    n_ops = fdp.ConsumeIntInRange(1, 12)

    # Pre-generate all (chunk, offset) pairs before spawning threads
    chunks: list[np.ndarray] = []
    offsets: list[int] = []
    for _ in range(n_ops):
        rows = fdp.ConsumeIntInRange(0, 16)
        offset = fdp.ConsumeIntInRange(0, rows + 3)  # offset may exceed len(chunk)
        if rows == 0 or action_dim == 0:
            chunk = np.zeros((rows, max(action_dim, 1)), dtype=np.float32)
        else:
            n_bytes = rows * action_dim * 4
            raw = fdp.ConsumeBytes(n_bytes)
            if len(raw) < n_bytes:
                raw = raw + b"\x00" * (n_bytes - len(raw))
            chunk = np.frombuffer(raw[:n_bytes], dtype=np.float32).copy().reshape(
                (rows, action_dim)
            )
        chunks.append(chunk)
        offsets.append(offset)

    thread_errors: list[Exception] = []

    def producer() -> None:
        for chunk, offset in zip(chunks, offsets):
            try:
                queue.push_chunk(chunk, offset=offset)
            except (ValueError, Exception) as exc:
                thread_errors.append(exc)

    def consumer() -> None:
        for _ in range(n_ops * 3):
            try:
                result = queue.pop()
                if result is not None:
                    if result.ndim != 1:
                        thread_errors.append(
                            AssertionError(
                                f"pop() returned {result.ndim}D array, expected 1D"
                            )
                        )
            except Exception as exc:  # noqa: BLE001
                thread_errors.append(exc)

    t_prod = threading.Thread(target=producer, name="FuzzProducer", daemon=True)
    t_cons = threading.Thread(target=consumer, name="FuzzConsumer", daemon=True)
    t_prod.start()
    t_cons.start()
    t_prod.join(timeout=3.0)
    t_cons.join(timeout=3.0)

    # Detect deadlocks: a thread still alive after the timeout is a hang, not a pass.
    if t_prod.is_alive() or t_cons.is_alive():
        raise AssertionError(
            "Thread deadlock detected — producer or consumer did not finish within 3 s"
        )

    # Re-raise the first error captured by either thread
    for exc in thread_errors:
        raise exc

    # Oracle: remaining count must be non-negative
    assert queue.remaining >= 0, (
        f"queue.remaining is negative: {queue.remaining}"
    )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
