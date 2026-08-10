# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fuzz StatsNormalizer — all four modes, extreme stat values (inf, nan, negative std),
arbitrary array shapes, passthrough-key preservation, and NaN propagation.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import atheris
import numpy as np

with atheris.instrument_imports():
    from physicalai.inference.preprocessors.stats_normalizer import StatsNormalizer

from _helpers import make_float_array, make_stats_dict

_MODES = ["mean_std", "min_max", "quantiles", "identity"]


def test_one_input(data: bytes) -> None:
    if len(data) < 8:
        return

    fdp = atheris.FuzzedDataProvider(data)
    mode = fdp.PickValueInList(_MODES)
    feature_name = fdp.ConsumeUnicodeNoSurrogates(32) or "observation.state"

    stat_dim = fdp.ConsumeIntInRange(1, 16)
    stats = make_stats_dict(fdp, feature_name, stat_dim=stat_dim, mode=mode)

    arr = make_float_array(fdp, max_ndim=3, max_dim=32)
    other_key = "passthrough_feature"
    other_arr = make_float_array(fdp, max_ndim=2, max_dim=16)

    inputs = {feature_name: arr, other_key: other_arr}

    # Check whether stats contain NaN (used by the propagation check below)
    stat_has_nan = any(
        not np.all(np.isfinite(v))
        for v in stats.get(feature_name, {}).values()
        if isinstance(v, np.ndarray)
    )

    try:
        normalizer = StatsNormalizer(mode=mode, features=[feature_name], stats=stats)
        outputs = normalizer(inputs)
    except (ValueError, TypeError, FloatingPointError):
        return

    assert other_key in outputs, (
        f"StatsNormalizer dropped key {other_key!r} which was not in features"
    )

    np.testing.assert_array_equal(
        outputs[other_key],
        other_arr,
        err_msg=f"StatsNormalizer modified non-listed key {other_key!r}",
    )

    if mode == "identity" and feature_name in outputs:
        np.testing.assert_array_equal(
            outputs[feature_name],
            arr,
            err_msg="identity mode should not modify the input array",
        )

    # NaN/Inf stats on a finite input must propagate — silent masking would hide
    # corrupted stats from downstream robot safety checks.
    if (
        mode != "identity"
        and stat_has_nan
        and arr.size > 0
        and np.all(np.isfinite(arr))
        and feature_name in outputs
        and outputs[feature_name].size > 0
    ):
        assert not np.all(np.isfinite(outputs[feature_name])), (
            f"StatsNormalizer silently produced finite output despite NaN/Inf stats "
            f"(mode={mode!r}).  NaN/Inf must propagate to be detectable downstream."
        )


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
