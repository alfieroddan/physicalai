# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz ActionChunkTrimmer — 3-D action output must be trimmed to n_action_steps; other keys pass through unchanged."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.postprocessors.action_chunk_trimmer import ActionChunkTrimmer
    from physicalai.inference.constants import ACTION

from _helpers import make_float_array


def test_one_input(data: bytes) -> None:
    if len(data) < 4:
        return

    fdp = atheris.FuzzedDataProvider(data)
    # Start from 0 — a zero n_action_steps must not crash (e.g. slice [:0] on action).
    n_action_steps = fdp.ConsumeIntInRange(0, 64)

    trimmer = ActionChunkTrimmer(n_action_steps=n_action_steps)

    include_action = fdp.ConsumeBool()
    action_ndim = fdp.ConsumeIntInRange(2, 3)  # 2-D or 3-D (with temporal axis)
    other_key = "extra_output"
    other_arr = make_float_array(fdp, max_ndim=2, max_dim=16)

    outputs: dict[str, np.ndarray] = {other_key: other_arr}

    if include_action:
        if action_ndim == 3:  # noqa: PLR2004
            B = fdp.ConsumeIntInRange(1, 4)
            T = fdp.ConsumeIntInRange(0, 128)
            A = fdp.ConsumeIntInRange(0, 32)
            action = np.zeros((B, T, A), dtype=np.float32)
            n_bytes = B * T * A * 4
            raw = fdp.ConsumeBytes(n_bytes)
            if len(raw) < n_bytes:
                raw = raw + b"\x00" * (n_bytes - len(raw))
            if n_bytes > 0:
                action = np.frombuffer(raw[:n_bytes], dtype=np.float32).copy().reshape((B, T, A))
        else:
            action = make_float_array(fdp, max_ndim=2, max_dim=32)
        outputs[ACTION] = action

    try:
        result = trimmer(dict(outputs))
    except KeyError:
        return  # expected when no "action" key

    if include_action and ACTION in result and result[ACTION].ndim == 3:  # noqa: PLR2004
        assert result[ACTION].shape[1] <= n_action_steps, (
            f"ActionChunkTrimmer did not trim: shape={result[ACTION].shape}, "
            f"n_action_steps={n_action_steps}"
        )

    assert other_key in result, f"ActionChunkTrimmer dropped key {other_key!r}"
    np.testing.assert_array_equal(
        result[other_key],
        other_arr,
        err_msg=f"ActionChunkTrimmer modified passthrough key {other_key!r}",
    )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
