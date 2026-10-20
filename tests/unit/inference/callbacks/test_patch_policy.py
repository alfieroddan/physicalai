# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Patch Policy temporal image windowing."""

from __future__ import annotations

from typing import Any

import numpy as np

from physicalai.inference.callbacks import Callback, PatchPolicyTemporalWindowCallback
from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.manifest import ComponentSpec


def _require_modified_inputs(inputs: dict[str, Any] | None) -> dict[str, Any]:
    """Return callback output after narrowing its optional contract.

    Raises:
        AssertionError: If the callback unexpectedly returns no modified inputs.
    """
    if inputs is None:
        msg = "Temporal window callback unexpectedly left single-frame inputs unchanged."
        raise AssertionError(msg)
    return inputs


def test_instantiates_from_registered_manifest_name() -> None:
    """The Patch Policy temporal callback resolves from exported manifests."""
    spec = ComponentSpec.model_validate({"type": "patch_policy_temporal_images", "window_size": 2, "stride": 1})

    callback = instantiate_component(Callback, spec)

    assert isinstance(callback, PatchPolicyTemporalWindowCallback)  # ruff: ignore[assert]


def test_builds_and_resets_camera_history() -> None:
    """The initial frame is repeated and reset clears prior episode history."""
    callback = PatchPolicyTemporalWindowCallback(window_size=2)
    first = np.ones((1, 3, 2, 2), dtype=np.float32)
    second = np.full((1, 3, 2, 2), 2.0, dtype=np.float32)

    initial = _require_modified_inputs(callback.on_predict_start({"images": {"front": first}}))
    updated = _require_modified_inputs(callback.on_predict_start({"images": {"front": second}}))

    np.testing.assert_allclose(initial["images"]["front"], np.stack([first, first], axis=1))
    np.testing.assert_allclose(updated["images"]["front"], np.stack([first, second], axis=1))

    callback.on_reset()
    reset = _require_modified_inputs(callback.on_predict_start({"images": {"front": second}}))
    np.testing.assert_allclose(reset["images"]["front"], np.stack([second, second], axis=1))
