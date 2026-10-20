# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy temporal image-window callback."""

from __future__ import annotations

from physicalai.config import export_config
from physicalai.inference.callbacks.rldx1_vtc import Rldx1VtcWindowCallback


@export_config(class_path="physicalai.inference.callbacks.PatchPolicyTemporalWindowCallback")
class PatchPolicyTemporalWindowCallback(Rldx1VtcWindowCallback):
    """Build Patch Policy camera history windows from single-frame observations.

    Args:
        window_size: Number of frames in each output window.
        stride: Environment-step stride between selected frames.
    """

    def __init__(self, window_size: int = 1, stride: int = 1) -> None:
        """Initialize the temporal window size and frame stride."""
        super().__init__(video_length=window_size, video_stride=stride)
        self._window_size = int(window_size)
        self._stride = int(stride)

    def __repr__(self) -> str:
        """Return the configured temporal window."""
        return f"{self.__class__.__name__}(window_size={self._window_size}, stride={self._stride})"


__all__ = ["PatchPolicyTemporalWindowCallback"]
