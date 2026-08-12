# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy postprocessor for MolmoAct2 exported models."""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from physicalai.inference.constants import ACTION
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.postprocessors.stats_denormalizer import StatsDenormalizer


class MolmoAct2Postprocessor(Postprocessor):
    """Clamp, denormalize, and optionally transform MolmoAct2 actions."""

    def __init__(
        self,
        *,
        action_stats: dict[str, Any] | None = None,
        adapt_to_so101: bool = False,
        joint_signs: list[float] | None = None,
        joint_offsets: list[float] | None = None,
    ) -> None:
        """Initialize the MolmoAct2 postprocessor.

        Args:
            action_stats: Quantile statistics used to denormalize actions.
            adapt_to_so101: Whether to transform actions to the SO-101 joint frame.
            joint_signs: Per-joint signs used by the SO-101 transform.
            joint_offsets: Per-joint offsets used by the SO-101 transform.

        Raises:
            ValueError: If ``joint_signs`` and ``joint_offsets`` have different lengths.
        """
        signs = joint_signs or []
        offsets = joint_offsets or []
        if len(signs) != len(offsets):
            msg = f"joint_signs ({len(signs)}) and joint_offsets ({len(offsets)}) must match"
            raise ValueError(msg)
        self._adapt_to_so101 = adapt_to_so101
        self._joint_signs = np.asarray(signs, dtype=np.float32)
        self._joint_offsets = np.asarray(offsets, dtype=np.float32)
        self._denormalizer = (
            StatsDenormalizer(stats={ACTION: action_stats}, mode="quantiles", features=[ACTION])
            if action_stats
            else None
        )

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = dict(outputs)
        action = result.get(ACTION, result.get("actions"))
        if action is None:
            msg = "MolmoAct2 postprocessor expected an action tensor"
            raise ValueError(msg)
        action = np.clip(np.asarray(action), -1.0, 1.0)
        if self._denormalizer is not None:
            action = self._denormalizer({ACTION: action})[ACTION]
        if self._adapt_to_so101:
            count = min(self._joint_signs.size, action.shape[-1])
            transformed = np.array(action, copy=True)
            transformed[..., :count] = self._joint_signs[:count] * (action[..., :count] - self._joint_offsets[:count])
            action = transformed
        result.pop("actions", None)
        result[ACTION] = action
        return result


__all__ = ["MolmoAct2Postprocessor"]
