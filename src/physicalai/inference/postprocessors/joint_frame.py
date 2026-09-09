# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Joint-frame action postprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from physicalai.inference.joint_transform import JointFrameTransform
from physicalai.inference.postprocessors.base import Postprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class JointFramePostprocessor(Postprocessor):
    """Map one output feature from checkpoint to robot joint coordinates."""

    def __init__(self, *, feature: str, signs: Sequence[float], offsets: Sequence[float]) -> None:
        """Configure the feature and calibration frame."""
        self._feature = feature
        self._transform = JointFrameTransform(signs=signs, offsets=offsets)

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform the configured feature while preserving all other outputs.

        Returns:
            A shallow copy with the transformed feature.

        Raises:
            ValueError: If the configured feature is absent.
        """
        if self._feature not in outputs:
            msg = f"Joint frame postprocessor expected feature {self._feature!r}"
            raise ValueError(msg)
        result = dict(outputs)
        result[self._feature] = self._transform.to_robot(np.asarray(outputs[self._feature]))
        return result


__all__ = ["JointFramePostprocessor"]
