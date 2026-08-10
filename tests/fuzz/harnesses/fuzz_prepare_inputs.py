# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz _prepare_inputs — dot-key collision consistency, filtering correctness, and no crashes.

When a flat key "obs.image" and a nested dict {"obs": {"image": x}} are both present,
the collision winner must be deterministic. This is a safety issue: an attacker
controlling the obs dict structure could silently substitute the wrong tensor.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.model import InferenceModel

from _helpers import make_float_array


def _call_prepare_inputs(
    inputs: dict,
    expected_keys: list[str] | None,
) -> dict:
    """Call InferenceModel._prepare_inputs via a lightweight mock self."""
    mock = types.SimpleNamespace()
    mock.adapter = types.SimpleNamespace(input_names=expected_keys or [])
    return InferenceModel._prepare_inputs(mock, inputs)  # type: ignore[arg-type]


@atheris.instrument_func
def _sub_collision_determinism(fdp: atheris.FuzzedDataProvider) -> None:
    """Assert that key-collision result is consistent regardless of dict ordering."""
    prefix = fdp.ConsumeUnicodeNoSurrogates(8) or "obs"
    suffix = fdp.ConsumeUnicodeNoSurrogates(8) or "image"

    if not prefix or not suffix:
        return

    dot_key = f"{prefix}.{suffix}"
    flat_val = make_float_array(fdp, max_ndim=2, max_dim=8)
    nested_val = make_float_array(fdp, max_ndim=2, max_dim=8)

    # Order A: flat key first, then nested dict
    inputs_a = {dot_key: flat_val, prefix: {suffix: nested_val}}
    # Order B: nested dict first, then flat key
    inputs_b = {prefix: {suffix: nested_val}, dot_key: flat_val}

    try:
        result_a = _call_prepare_inputs(inputs_a, [dot_key])
        result_b = _call_prepare_inputs(inputs_b, [dot_key])
    except KeyError:
        return  # key may not survive the filter
    except ValueError:
        return  # collision correctly rejected — expected outcome per I-7

    # Neither call raised: collision was silently resolved, which is a regression.
    raise AssertionError(
        f"_prepare_inputs silently resolved flat+nested collision for key {dot_key!r} "
        f"without raising ValueError; regression against I-7"
    )


@atheris.instrument_func
def _sub_no_crash(fdp: atheris.FuzzedDataProvider) -> None:
    """No crash for arbitrary key names and structures."""
    n_keys = fdp.ConsumeIntInRange(0, 6)
    inputs: dict = {}
    for _ in range(n_keys):
        key = fdp.ConsumeUnicodeNoSurrogates(24)
        if not key:
            continue
        if fdp.ConsumeBool():
            # Nested dict
            sub_key = fdp.ConsumeUnicodeNoSurrogates(16) or "x"
            inputs[key] = {sub_key: make_float_array(fdp, max_ndim=2, max_dim=8)}
        else:
            inputs[key] = make_float_array(fdp, max_ndim=2, max_dim=8)

    expected = None
    if fdp.ConsumeBool() and inputs:
        # Pick a random subset of flat-projected keys as expected
        flat_keys = []
        for k, v in inputs.items():
            if isinstance(v, dict):
                for sk in v:
                    flat_keys.append(f"{k}.{sk}")
            else:
                flat_keys.append(k)
        if flat_keys:
            n = fdp.ConsumeIntInRange(1, min(len(flat_keys), 4))
            expected = flat_keys[:n]

    try:
        result = _call_prepare_inputs(inputs, expected)
    except (KeyError, ValueError):
        return  # Acceptable — missing key or collision between flat/nested key

    # Oracle: only applies when flattening ran (passthrough returns nested dicts as-is)
    if expected is not None:
        for v in result.values():
            assert isinstance(v, np.ndarray), (
                f"_prepare_inputs returned a non-ndarray value: {type(v).__name__}"
            )


@atheris.instrument_func
def _sub_filter_exact(fdp: atheris.FuzzedDataProvider) -> None:
    """With a single expected key present, output has exactly that key."""
    key = fdp.ConsumeUnicodeNoSurrogates(16) or "state"
    arr = make_float_array(fdp, max_ndim=2, max_dim=8)
    inputs = {key: arr}

    try:
        result = _call_prepare_inputs(inputs, [key])
    except KeyError:
        return

    assert list(result.keys()) == [key], (
        f"_prepare_inputs with expected=[{key!r}] returned keys {list(result.keys())!r}"
    )
    np.testing.assert_array_equal(
        result[key],
        arr,
        err_msg=f"_prepare_inputs mutated value for key {key!r}",
    )


def test_one_input(data: bytes) -> None:
    if len(data) < 4:
        return

    fdp = atheris.FuzzedDataProvider(data)
    sub = fdp.ConsumeIntInRange(0, 2)

    if sub == 0:
        _sub_collision_determinism(fdp)
    elif sub == 1:
        _sub_no_crash(fdp)
    else:
        _sub_filter_exact(fdp)


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
