# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit layouts for batched image arrays."""

from enum import StrEnum


class ImageLayout(StrEnum):
    """Axis order of a batched image array.

    Attributes:
        BCHW: Batch, channels, height, width (channels-first).
        BHWC: Batch, height, width, channels (channels-last).
    """

    BCHW = "BCHW"
    BHWC = "BHWC"


def infer_image_layout(shape: tuple[int, ...]) -> ImageLayout:
    """Infer the layout of a 4D image using the legacy heuristic.

    Args:
        shape: Shape of an image array already validated as four-dimensional.

    Returns:
        Inferred BCHW or BHWC layout.

    Raises:
        ValueError: If both candidate channel axes look like channel counts.
    """
    channel_counts = {1, 2, 3, 4}

    if shape[1] in channel_counts and shape[-1] in channel_counts:
        msg = (
            f"ambiguous layout: both dim 1 ({shape[1]}) and dim -1 ({shape[-1]}) "
            "look like standard channel counts; provide input with spatial dims > 4"
        )
        raise ValueError(msg)

    if shape[-1] in channel_counts and shape[1] not in channel_counts:
        return ImageLayout.BHWC

    return ImageLayout.BCHW
