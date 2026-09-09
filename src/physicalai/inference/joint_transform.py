# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint calibration frame transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class JointFrameTransform:
    """Map leading joint values between robot and checkpoint calibration frames."""

    def __init__(self, *, signs: Sequence[float], offsets: Sequence[float]) -> None:
        """Store the joint signs and offsets.

        Raises:
            ValueError: If signs and offsets differ in length or a sign is not +/-1.
        """
        if len(signs) != len(offsets):
            msg = f"signs ({len(signs)}) and offsets ({len(offsets)}) must match"
            raise ValueError(msg)
        if any(sign not in {-1.0, 1.0} for sign in signs):
            msg = "Joint frame transform signs must be either -1 or 1."
            raise ValueError(msg)
        self._signs = np.asarray(signs, dtype=np.float32)
        self._offsets = np.asarray(offsets, dtype=np.float32)

    def to_checkpoint(self, values: np.ndarray) -> np.ndarray:
        """Map robot-frame joints to the checkpoint frame.

        Returns:
            A transformed copy of ``values``.
        """
        return self._apply(values, inverse=False)

    def to_robot(self, values: np.ndarray) -> np.ndarray:
        """Map checkpoint-frame joints back to the robot frame.

        Returns:
            A transformed copy of ``values``.
        """
        return self._apply(values, inverse=True)

    def _apply(self, values: np.ndarray, *, inverse: bool) -> np.ndarray:
        count = min(self._signs.size, values.shape[-1])
        output = np.array(values, copy=True)
        joints = values[..., :count]
        output[..., :count] = (
            self._signs[:count] * (joints - self._offsets[:count])
            if inverse
            else self._signs[:count] * joints + self._offsets[:count]
        )
        return output
