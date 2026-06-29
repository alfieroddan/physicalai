# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 inference postprocessor."""

from __future__ import annotations

from typing import Any

import numpy as np

from physicalai.inference.constants import ACTION

from .base import Postprocessor

_EPS = 1e-8


class MolmoAct2Postprocessor(Postprocessor):
    """Map model outputs to ``action`` and restore action-space scaling."""

    def __init__(
        self,
        *,
        action_key: str | None = None,
        env_action_dim: int | None = None,
        action_stats: dict[str, list[float] | np.ndarray] | None = None,
    ) -> None:
        self._action_key = action_key
        self._env_action_dim = int(env_action_dim) if env_action_dim is not None else None

        self._q01: np.ndarray | None = None
        self._q99: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        if action_stats is not None:
            q01 = action_stats.get("q01")
            q99 = action_stats.get("q99")
            if q01 is not None and q99 is not None:
                self._q01 = np.asarray(q01, dtype=np.float32)
                self._q99 = np.asarray(q99, dtype=np.float32)
            mask = action_stats.get("mask")
            if mask is not None:
                self._mask = np.asarray(mask, dtype=bool)

    def _resolve_action_key(self, outputs: dict[str, np.ndarray]) -> str:
        if ACTION in outputs:
            return ACTION
        if self._action_key is not None:
            return self._action_key
        if "actions" in outputs:
            return "actions"
        return next(iter(outputs))

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        if self._q01 is None or self._q99 is None:
            return action
        denom = self._q99 - self._q01
        denom = np.where(denom == 0, _EPS, denom)
        denorm = (action + 1.0) * denom / 2.0 + self._q01
        if self._mask is not None:
            mask = self._mask
            while mask.ndim < denorm.ndim:
                mask = np.expand_dims(mask, axis=0)
            denorm = np.where(mask, denorm, action)
        return denorm

    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = dict(outputs)
        action_key = self._resolve_action_key(result)
        action = np.asarray(result.pop(action_key))

        if self._env_action_dim is not None:
            action = action[..., : self._env_action_dim]

        action = np.clip(action, -1.0, 1.0)
        action = self._denormalize(action)
        result[ACTION] = action.astype(np.float32)
        return result


__all__ = ["MolmoAct2Postprocessor"]
