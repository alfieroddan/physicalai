# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images for SmolVLA."""

from __future__ import annotations

import operator

import cv2
import numpy as np

from physicalai.inference.constants import IMAGE_MASKS, IMAGES

from .base import Preprocessor


class ResizeSmolVLA(Preprocessor):
    """Preprocessor for SmolVLA inference image remap/resize/normalize/padding.

    This preprocessor handles:
    - Remapping image keys via dictionary mapping
    - Resizing images to specified resolution with aspect ratio preservation
    - Normalizing pixel values to [-1, 1] range
    - Handling missing cameras with empty placeholders
    - Generating image masks for valid/padded regions
    """

    _VIDEO_IMAGE_DIMS = 5
    _BATCHED_IMAGE_DIMS = 4

    def __init__(
        self,
        image_resolution: tuple[int, int] = (512, 512),
        image_key_rename_map: dict[str, str] | None = None,
        image_features: list[str] | None = None,
        empty_cameras: int = 0,
    ) -> None:
        """Initialize the SmolVLA preprocessor with remapping and masking.

        Args:
            image_resolution: Target resolution as (height, width). Defaults to (512, 512).
            image_key_rename_map: Mapping from source image keys to target camera names.
            image_features: Expected list of camera names in output. If empty,
                uses values from image_key_rename_map.
            empty_cameras: Number of missing cameras to fill with empty (padded) images.
        """
        super().__init__()
        self.image_resolution = image_resolution
        self.image_key_rename_map = image_key_rename_map or {}
        self.image_features = image_features or []
        self.empty_cameras = empty_cameras

    EXTRA = "extra"

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and prepare images for model inference.

        Remaps image keys, resizes with aspect ratio preservation, normalizes
        to [-1, 1], and fills missing cameras with empty placeholders.

        Returns:
            Updated inputs containing processed image tensors and masks.
        """
        inputs = dict(inputs)

        image_items = self._collect_image_items(inputs)

        keyed_images: list[tuple[str, np.ndarray, np.ndarray]] = []
        for key, source_img in image_items:
            camera_key = self._map_image_key(key)
            input_img = source_img[:, -1, :, :, :] if source_img.ndim == self._VIDEO_IMAGE_DIMS else source_img
            resized_img = self._resize_with_pad(input_img, *self.image_resolution, pad_value=0)
            resized_img = resized_img * 2.0 - 1.0
            mask = np.ones(resized_img.shape[0], dtype=np.bool_)
            keyed_images.append((camera_key, resized_img, mask))

        keyed_images.sort(key=operator.itemgetter(0))

        expected_targets = [self._normalize_camera_name(name) for name in self.image_features]
        if not expected_targets:
            expected_targets = [self._normalize_camera_name(name) for name in self.image_key_rename_map.values()]

        present_targets = {name for name, _, _ in keyed_images}
        missing_expected = [name for name in expected_targets if name not in present_targets]
        num_empty_cameras = min(self.empty_cameras, len(missing_expected))

        if num_empty_cameras > 0 and keyed_images:
            for target in missing_expected[:num_empty_cameras]:
                keyed_images.append(
                    (
                        target,
                        np.full_like(keyed_images[-1][1], -1.0),
                        np.zeros_like(keyed_images[-1][2], dtype=np.bool_),
                    ),
                )

        keyed_images.sort(key=operator.itemgetter(0))

        inputs[IMAGES] = {name: img for name, img, _ in keyed_images}
        inputs[IMAGE_MASKS] = {name: mask for name, _, mask in keyed_images}

        extra = inputs.get(self.EXTRA)
        if not isinstance(extra, dict):
            extra = {}
        for name, _, mask in keyed_images:
            extra[f"{IMAGES}.{name}_padding_mask"] = mask
        inputs[self.EXTRA] = extra

        return inputs

    @staticmethod
    def _collect_image_items(inputs: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
        """Collect all image items from the input dict.

        Returns:
            List of (key, image) tuples.
        """
        if IMAGES in inputs and isinstance(inputs[IMAGES], dict):
            return [(str(name), img) for name, img in inputs[IMAGES].items()]

        img_keys = [key for key in inputs if key.startswith(f"{IMAGES}.")]
        return [(key, inputs[key]) for key in img_keys]

    def _map_image_key(self, key: str) -> str:
        """Map source image key to target camera name.

        Returns:
            Normalized target camera name.
        """
        suffix = key.removeprefix(f"{IMAGES}.")
        mapped = (
            self.image_key_rename_map.get(key)
            or self.image_key_rename_map.get(f"observation.{IMAGES}.{suffix}")
            or self.image_key_rename_map.get(f"{IMAGES}.{suffix}")
            or self.image_key_rename_map.get(suffix)
            or suffix
        )
        return self._normalize_camera_name(str(mapped))

    @staticmethod
    def _normalize_camera_name(name: str) -> str:
        """Normalize camera name by removing known prefixes.

        Returns:
            Camera name without any images/observation prefix.
        """
        if name.startswith(f"{IMAGES}."):
            return name[len(f"{IMAGES}.") :]
        if name.startswith(f"observation.{IMAGES}."):
            return name[len(f"observation.{IMAGES}.") :]
        return name

    @staticmethod
    def _resize_with_pad(img: np.ndarray, width: int, height: int, pad_value: int = -1) -> np.ndarray:
        """Resize image with aspect ratio preservation via padding.

        Returns:
            Resized and padded image batch with shape (B, C, height, width).

        Raises:
            ValueError: If input does not have shape (B, C, H, W).
        """
        if img.ndim != ResizeSmolVLA._BATCHED_IMAGE_DIMS:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        batch = []
        for i in range(img.shape[0]):
            hwc = np.transpose(img[i], (1, 2, 0))
            resized_hwc = cv2.resize(hwc, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            batch.append(np.transpose(resized_hwc, (2, 0, 1)))
        resized_img = np.stack(batch, axis=0)

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        if pad_height > 0 or pad_width > 0:
            padded = np.full(
                (resized_img.shape[0], resized_img.shape[1], resized_height + pad_height, resized_width + pad_width),
                fill_value=pad_value,
                dtype=resized_img.dtype,
            )
            padded[:, :, pad_height:, pad_width:] = resized_img
            return padded
        return resized_img
