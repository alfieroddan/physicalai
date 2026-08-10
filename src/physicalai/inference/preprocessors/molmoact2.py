# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy preprocessors for MolmoAct2 exported models."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np
from typing_extensions import override

from physicalai.inference.constants import IMAGES, STATE, TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.preprocessors.stats_normalizer import StatsNormalizer

_STATE_START_TOKEN = "<state_start>"  # noqa: S105
_STATE_END_TOKEN = "<state_end>"  # noqa: S105
_STATE_TOKEN_PREFIX = "<state_"  # noqa: S105
_ACTION_OUTPUT_TOKEN = "<action_output>"  # noqa: S105
_TRAILING_PUNCTUATION = ".,!?;:"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)
_IMAGE_NDIM = 4
_PACKED_IMAGE_NDIM = 5
_NUM_CHANNELS = 3


def _joint_transform(
    values: np.ndarray,
    signs: np.ndarray,
    offsets: np.ndarray,
    *,
    inverse: bool,
) -> np.ndarray:
    """Apply the MolmoAct2 joint-frame transform to leading dimensions."""
    count = min(signs.size, values.shape[-1])
    output = np.array(values, copy=True)
    joints = values[..., :count]
    output[..., :count] = signs[:count] * (joints - offsets[:count]) if inverse else signs[:count] * joints + offsets[:count]
    return output


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    for pattern in _PREFIX_PATTERNS:
        normalized = pattern.sub("", normalized, count=1).strip()
    return normalized.rstrip(_TRAILING_PUNCTUATION).strip().lower()


def _discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    values = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    values = np.clip(values, -1.0, 1.0)
    token_ids = np.rint((values + 1.0) / 2.0 * (num_state_tokens - 1)).astype(np.int64)
    payload = "".join(f"{_STATE_TOKEN_PREFIX}{int(token_id)}>" for token_id in token_ids.reshape(-1))
    return f"{_STATE_START_TOKEN}{payload}{_STATE_END_TOKEN}"


def _wrapped_text(value: str, start: str, end: str, *, enabled: bool) -> str:
    if not value or not enabled or (value.startswith(start) and value.endswith(end)):
        return value
    return f"{start}{value}{end}"


def _robot_prompt(
    *,
    task: str,
    state: np.ndarray,
    num_state_tokens: int,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup = _wrapped_text(setup_type, "<setup_start>", "<setup_end>", enabled=add_setup_tokens)
    control = _wrapped_text(control_mode, "<control_start>", "<control_end>", enabled=add_control_tokens)
    discrete_state = _discrete_state_string(state, num_state_tokens)
    prompt = (
        f"The task is to {task}. The setup is {setup}. "
        f"The current state of the robot is {discrete_state}. "
        f"The expected control mode is {control}. "
        "Given these, what action should the robot take to complete the task?"
    )
    if num_images == 1:
        image_prefix = "<|image|>"
    elif num_images > 1:
        image_prefix = "".join(f"Image {index + 1}<|image|>" for index in range(num_images))
    else:
        image_prefix = ""
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{_ACTION_OUTPUT_TOKEN}"


class MolmoAct2Preprocessor(Preprocessor):
    """Prepare MolmoAct2 prompts and images before tokenization."""

    def __init__(
        self,
        *,
        image_keys: list[str],
        state_stats: dict[str, Any] | None = None,
        image_size: tuple[int, int] = (378, 378),
        num_state_tokens: int = 256,
        setup_type: str = "",
        control_mode: str = "",
        add_setup_tokens: bool = True,
        add_control_tokens: bool = True,
        adapt_to_so101: bool = False,
        joint_signs: list[float] | None = None,
        joint_offsets: list[float] | None = None,
    ) -> None:
        if num_state_tokens <= 0:
            msg = f"num_state_tokens must be > 0, got {num_state_tokens}"
            raise ValueError(msg)
        signs = joint_signs or []
        offsets = joint_offsets or []
        if len(signs) != len(offsets):
            msg = f"joint_signs ({len(signs)}) and joint_offsets ({len(offsets)}) must match"
            raise ValueError(msg)

        self._image_keys = list(image_keys)
        self._image_size = tuple(image_size)
        self._num_state_tokens = num_state_tokens
        self._setup_type = setup_type
        self._control_mode = control_mode
        self._add_setup_tokens = add_setup_tokens
        self._add_control_tokens = add_control_tokens
        self._adapt_to_so101 = adapt_to_so101
        self._joint_signs = np.asarray(signs, dtype=np.float32)
        self._joint_offsets = np.asarray(offsets, dtype=np.float32)
        self._normalizer = (
            StatsNormalizer(stats={STATE: state_stats}, mode="quantiles", features=[STATE]) if state_stats else None
        )

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        outputs = dict(inputs)
        state = outputs.get(STATE, outputs.get(f"observation.{STATE}"))
        if state is None:
            msg = f"MolmoAct2 requires {STATE!r} in its input"
            raise ValueError(msg)
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if self._adapt_to_so101:
            state = _joint_transform(state, self._joint_signs, self._joint_offsets, inverse=False)
        if self._normalizer is not None:
            state = self._normalizer({STATE: state})[STATE]
        state = np.clip(state, -1.0, 1.0)

        images = self._extract_images(outputs, batch_size=state.shape[0])
        tasks = self._extract_tasks(outputs, batch_size=state.shape[0])
        outputs[IMAGES] = np.stack([self._resize_image(image) for image in images], axis=0)
        outputs[TASK] = [
            _robot_prompt(
                task=tasks[index],
                state=state[index],
                num_state_tokens=self._num_state_tokens,
                setup_type=self._setup_type,
                control_mode=self._control_mode,
                add_setup_tokens=self._add_setup_tokens,
                add_control_tokens=self._add_control_tokens,
                num_images=len(images),
            )
            for index in range(state.shape[0])
        ]
        outputs.pop(STATE, None)
        outputs.pop(f"observation.{STATE}", None)
        return outputs

    def _extract_images(self, inputs: dict[str, Any], *, batch_size: int) -> list[np.ndarray]:
        images_value = inputs.get(IMAGES)
        images: list[np.ndarray] = []
        if self._image_keys:
            for name in self._image_keys:
                flat_key = name if name.startswith(f"{IMAGES}.") else f"{IMAGES}.{name}"
                if flat_key in inputs:
                    images.append(np.asarray(inputs[flat_key]))
                elif isinstance(images_value, dict) and name.removeprefix(f"{IMAGES}.") in images_value:
                    images.append(np.asarray(images_value[name.removeprefix(f"{IMAGES}.")]))
        elif isinstance(images_value, np.ndarray):
            images = [images_value]
        elif isinstance(images_value, dict):
            images = [np.asarray(value) for key, value in images_value.items() if "is_pad" not in str(key)]
        else:
            keys = sorted(key for key in inputs if key.startswith(f"{IMAGES}.") and "is_pad" not in key)
            images = [np.asarray(inputs[key]) for key in keys]

        if not images:
            msg = "MolmoAct2 requires at least one image input"
            raise ValueError(msg)
        for image in images:
            if image.ndim != _IMAGE_NDIM or image.shape[1] != _NUM_CHANNELS:
                msg = f"Expected BCHW image with 3 channels, got {image.shape}"
                raise ValueError(msg)
            if image.shape[0] != batch_size:
                msg = f"Image batch size mismatch: expected {batch_size}, got {image.shape[0]}"
                raise ValueError(msg)
        return images

    @staticmethod
    def _extract_tasks(inputs: dict[str, Any], *, batch_size: int) -> list[str]:
        source = inputs.get(TASK, inputs.get(f"observation.{TASK}", inputs.get("observation.language")))
        if source is None:
            msg = f"MolmoAct2 requires {TASK!r} in its input"
            raise ValueError(msg)
        if isinstance(source, str):
            tasks = [source] * batch_size
        else:
            tasks = [str(value) for value in np.asarray(source).reshape(-1).tolist()]
            if len(tasks) == 1 and batch_size > 1:
                tasks *= batch_size
        if len(tasks) != batch_size:
            msg = f"Expected {batch_size} task strings, got {len(tasks)}"
            raise ValueError(msg)
        return [_normalize_text(task) for task in tasks]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = self._image_size
        output: list[np.ndarray] = []
        for sample in image:
            if sample.dtype == np.uint8:
                pixels = sample
            elif np.issubdtype(sample.dtype, np.floating):
                float_pixels = sample.astype(np.float32)
                if float(np.max(float_pixels)) <= 1.0:
                    float_pixels *= 255.0
                pixels = np.clip(float_pixels, 0.0, 255.0).astype(np.uint8)
            else:
                msg = f"Unsupported image dtype: {sample.dtype}"
                raise ValueError(msg)
            hwc = np.transpose(pixels, (1, 2, 0))
            resized = cv2.resize(hwc, (width, height), interpolation=cv2.INTER_LINEAR_EXACT)
            output.append(np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0)
        return np.stack(output, axis=0)


class MolmoAct2ModelInputs(Preprocessor):
    """Assemble tokenized prompts and packed images into MolmoAct2 model inputs."""

    def __init__(
        self,
        *,
        max_action_dim: int,
        action_dim: int,
        bos_token_id: int,
        pad_token_id: int,
        image_placeholder_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        image_patch_id: int,
        image_col_id: int | None,
        low_res_image_start_token_id: int | None,
        image_size: tuple[int, int] = (378, 378),
        patch_size: int = 14,
        pooling_size: tuple[int, int] = (2, 2),
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        image_use_col_tokens: bool = True,
        use_single_crop_col_tokens: bool = False,
        use_single_crop_start_token: bool = True,
        image_token_ids: list[int] | None = None,
    ) -> None:
        self._max_action_dim = max_action_dim
        self._action_dim = action_dim
        self._bos_token_id = bos_token_id
        self._pad_token_id = pad_token_id
        self._placeholder_id = image_placeholder_token_id
        self._image_start_id = image_start_token_id
        self._image_end_id = image_end_token_id
        self._image_patch_id = image_patch_id
        self._image_col_id = image_col_id
        self._low_res_start_id = low_res_image_start_token_id or image_start_token_id
        self._height, self._width = image_size
        self._patch_size = patch_size
        self._pool_h, self._pool_w = pooling_size
        self._mean = np.asarray(image_mean or [0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.asarray(image_std or [0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
        self._image_use_col_tokens = image_use_col_tokens
        self._use_single_crop_col_tokens = use_single_crop_col_tokens
        self._use_single_crop_start_token = use_single_crop_start_token
        self._image_token_ids = np.asarray(image_token_ids or [], dtype=np.int64)
        self._pooling, self._pooled_h, self._pooled_w = self._pooling_indices()

    @override
    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        input_ids = np.asarray(inputs[TOKENIZED_PROMPT], dtype=np.int64)
        attention_mask = np.asarray(inputs[TOKENIZED_PROMPT_MASK], dtype=np.int64)
        input_ids, attention_mask = self._insert_bos(input_ids, attention_mask)

        images = np.asarray(inputs[IMAGES], dtype=np.float32)
        if images.ndim != _PACKED_IMAGE_NDIM:
            msg = f"Expected packed images (N, B, C, H, W), got {images.shape}"
            raise ValueError(msg)
        num_images, batch_size, channels, height, width = images.shape
        if channels != _NUM_CHANNELS or (height, width) != (self._height, self._width):
            msg = f"Unexpected packed image shape {images.shape}"
            raise ValueError(msg)

        flat_images = images.transpose(1, 0, 2, 3, 4).reshape(batch_size * num_images, channels, height, width)
        pixel_values = self._patchify((flat_images - self._mean) / self._std)
        grids = np.tile(np.array([[self._pooled_h, self._pooled_w, 0, 0]], dtype=np.int64), (batch_size * num_images, 1))
        input_ids, attention_mask = self._expand_placeholders(input_ids, attention_mask, grids)
        token_type_ids = self._token_type_ids(input_ids, attention_mask)
        batched_images = pixel_values.reshape(batch_size, num_images, pixel_values.shape[1], pixel_values.shape[2])

        pooling = []
        patches_per_image = pixel_values.shape[1]
        for image_index in range(num_images):
            block = np.where(self._pooling >= 0, self._pooling + image_index * patches_per_image, self._pooling)
            pooling.append(block)
        token_pooling = np.tile(np.concatenate(pooling, axis=0)[None, ...], (batch_size, 1, 1))
        action_dim_is_pad = np.ones((batch_size, self._max_action_dim), dtype=np.bool_)
        action_dim_is_pad[:, : self._action_dim] = False

        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            IMAGES: batched_images.astype(np.float32),
            "token_pooling": token_pooling.astype(np.int64),
            "action_dim_is_pad": action_dim_is_pad,
        }
        if token_type_ids is not None:
            outputs["token_type_ids"] = token_type_ids
        return outputs

    def _insert_bos(self, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        for row_ids, row_mask in zip(ids, mask, strict=True):
            valid_ids = row_ids[row_mask.astype(np.bool_)]
            if valid_ids.size == 0 or valid_ids[0] != self._bos_token_id:
                valid_ids = np.concatenate((np.array([self._bos_token_id], dtype=ids.dtype), valid_ids))
            rows.append(valid_ids)
        width = max((row.size for row in rows), default=1)
        output_ids = np.full((len(rows), width), self._pad_token_id, dtype=ids.dtype)
        output_mask = np.zeros((len(rows), width), dtype=mask.dtype)
        for index, row in enumerate(rows):
            output_ids[index, : row.size] = row
            output_mask[index, : row.size] = 1
        return output_ids, output_mask

    def _patchify(self, pixels: np.ndarray) -> np.ndarray:
        count, channels, height, width = pixels.shape
        patch = self._patch_size
        if height % patch or width % patch:
            msg = f"Image size {(height, width)} must be divisible by patch_size={patch}"
            raise ValueError(msg)
        pixels = pixels.transpose(0, 2, 3, 1)
        pixels = pixels.reshape(count, height // patch, patch, width // patch, patch, channels)
        return pixels.transpose(0, 1, 3, 2, 4, 5).reshape(count, -1, patch * patch * channels)

    def _pooling_indices(self) -> tuple[np.ndarray, int, int]:
        patch_h = self._height // self._patch_size
        patch_w = self._width // self._patch_size
        pooled_h = (patch_h + self._pool_h - 1) // self._pool_h
        pooled_w = (patch_w + self._pool_w - 1) // self._pool_w
        pad_h = pooled_h * self._pool_h - patch_h
        pad_w = pooled_w * self._pool_w - patch_w
        indices = np.arange(patch_h * patch_w, dtype=np.int64).reshape(patch_h, patch_w)
        indices = np.pad(
            indices,
            ((pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2)),
            constant_values=-1,
        )
        pooling = indices.reshape(pooled_h, self._pool_h, pooled_w, self._pool_w)
        return pooling.transpose(0, 2, 1, 3).reshape(-1, self._pool_h * self._pool_w), pooled_h, pooled_w

    def _image_sequence(self, grid: np.ndarray) -> list[int]:
        resized_h, resized_w, height, width = (int(value) for value in grid)

        def rows(row_count: int, col_count: int, *, use_col: bool) -> list[int]:
            row = [self._image_patch_id] * col_count
            if use_col and self._image_col_id is not None:
                row.append(self._image_col_id)
            return row * row_count

        if height == 0 or width == 0:
            return [
                self._image_start_id,
                *rows(resized_h, resized_w, use_col=self._use_single_crop_col_tokens),
                self._image_end_id,
            ]
        low_start = self._low_res_start_id if self._use_single_crop_start_token else self._image_start_id
        return [
            low_start,
            *rows(resized_h, resized_w, use_col=self._use_single_crop_col_tokens),
            self._image_end_id,
            self._image_start_id,
            *rows(height, width, use_col=self._image_use_col_tokens),
            self._image_end_id,
        ]

    def _expand_placeholders(
        self,
        ids: np.ndarray,
        mask: np.ndarray,
        grids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows: list[np.ndarray] = []
        grid_index = 0
        for row_ids, row_mask in zip(ids, mask, strict=True):
            expanded: list[int] = []
            for token in row_ids[row_mask.astype(np.bool_)]:
                if int(token) == self._placeholder_id:
                    if grid_index >= grids.shape[0]:
                        msg = "Not enough image grids to expand all <|image|> placeholders"
                        raise ValueError(msg)
                    expanded.extend(self._image_sequence(grids[grid_index]))
                    grid_index += 1
                else:
                    expanded.append(int(token))
            rows.append(np.asarray(expanded, dtype=ids.dtype))
        if grid_index != grids.shape[0]:
            msg = f"Image placeholders ({grid_index}) do not match images ({grids.shape[0]})"
            raise ValueError(msg)
        width = max((row.size for row in rows), default=1)
        output_ids = np.full((len(rows), width), self._pad_token_id, dtype=ids.dtype)
        output_mask = np.zeros((len(rows), width), dtype=mask.dtype)
        for index, row in enumerate(rows):
            output_ids[index, : row.size] = row
            output_mask[index, : row.size] = 1
        return output_ids, output_mask

    def _token_type_ids(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if self._image_token_ids.size == 0:
            return None
        return (np.isin(ids, self._image_token_ids) & mask.astype(np.bool_)).astype(np.int64)


__all__ = ["MolmoAct2ModelInputs", "MolmoAct2Preprocessor"]
