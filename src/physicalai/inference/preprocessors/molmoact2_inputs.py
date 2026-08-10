# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NumPy model-input assembly for MolmoAct2 inference.

Mirrors the PyTorch ``build_model_inputs`` used during training/export so the
exported OpenVINO graph receives identical, fully-prepared tensors. Turns a
tokenized prompt (with ``<|image|>`` placeholders) and patchified images into
``input_ids`` (placeholders expanded), ``attention_mask``, ``token_type_ids``,
per-example batched ``images``, ``token_pooling`` and ``action_dim_is_pad``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MolmoAct2InputConfig:
    """Token ids and layout flags needed to assemble MolmoAct2 model inputs."""

    image_placeholder_token_id: int
    image_patch_id: int
    image_start_token_id: int
    image_end_token_id: int
    image_col_id: int | None = None
    low_res_image_start_token_id: int | None = None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_use_col_tokens: bool = True
    use_single_crop_col_tokens: bool = False
    use_single_crop_start_token: bool = True
    max_action_dim: int = 32
    env_action_dim: int = 0
    _image_token_ids: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        ids = [
            self.image_patch_id,
            self.image_col_id,
            self.image_start_token_id,
            self.low_res_image_start_token_id,
            self.frame_start_token_id,
            self.image_end_token_id,
            self.frame_end_token_id,
            self.image_low_res_id,
        ]
        self._image_token_ids = [int(token_id) for token_id in ids if token_id is not None]

    @property
    def image_token_ids(self) -> list[int]:
        """Token ids that mark image content (for token type ids)."""
        return self._image_token_ids


def _image_token_ids_for_grid(config: MolmoAct2InputConfig, grid: np.ndarray) -> list[int]:
    """Expand a single image grid into its sequence of image token ids."""
    resized_h, resized_w, height, width = (int(x) for x in np.asarray(grid).reshape(-1)[:4].tolist())

    image_patch_id = int(config.image_patch_id)
    image_start_token_id = int(config.image_start_token_id)
    image_end_token_id = int(config.image_end_token_id)
    image_col_id = None if config.image_col_id is None else int(config.image_col_id)
    low_res_start_id = (
        int(config.low_res_image_start_token_id)
        if config.low_res_image_start_token_id is not None
        else image_start_token_id
    )

    image_use_col_tokens = bool(config.image_use_col_tokens)
    use_single_crop_col_tokens = (
        image_use_col_tokens if config.use_single_crop_col_tokens is None else bool(config.use_single_crop_col_tokens)
    )
    use_single_crop_start_token = bool(config.use_single_crop_start_token)

    def make_rows(num_rows: int, num_cols: int, *, use_col: bool) -> list[int]:
        row = [image_patch_id] * num_cols
        if use_col and image_col_id is not None:
            row += [image_col_id]
        return row * num_rows

    if height == 0 or width == 0:
        return [
            image_start_token_id,
            *make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens),
            image_end_token_id,
        ]

    high_res = [image_start_token_id, *make_rows(height, width, use_col=image_use_col_tokens), image_end_token_id]
    low_start = low_res_start_id if use_single_crop_start_token else image_start_token_id
    low_res = [low_start, *make_rows(resized_h, resized_w, use_col=use_single_crop_col_tokens), image_end_token_id]
    return low_res + high_res


def _build_token_type_ids(
    config: MolmoAct2InputConfig, input_ids: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray | None:
    """Mark image tokens (1) vs. text tokens (0), respecting the attention mask."""
    image_token_ids = config.image_token_ids
    if not image_token_ids:
        return None
    token_set = np.asarray(image_token_ids, dtype=input_ids.dtype)
    is_image = np.isin(input_ids, token_set).astype(np.int64)
    return is_image * attention_mask.astype(np.int64)


def expand_image_placeholders(
    *,
    config: MolmoAct2InputConfig,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Replace each ``<|image|>`` placeholder with its expanded image token ids."""
    if int(image_grids.shape[0]) == 0:
        return input_ids, attention_mask, _build_token_type_ids(config, input_ids, attention_mask)

    pad_values = input_ids[attention_mask == 0]
    pad_token_id = int(pad_values[0]) if pad_values.size > 0 else 0
    placeholder_id = int(config.image_placeholder_token_id)

    expanded_rows: list[list[int]] = []
    grid_idx = 0
    for batch_idx in range(int(input_ids.shape[0])):
        valid = attention_mask[batch_idx].astype(bool)
        expanded: list[int] = []
        for token in input_ids[batch_idx][valid].tolist():
            token_int = int(token)
            if token_int == placeholder_id:
                if grid_idx >= int(image_grids.shape[0]):
                    msg = "Not enough image grids to expand all <|image|> placeholders."
                    raise ValueError(msg)
                expanded.extend(_image_token_ids_for_grid(config, image_grids[grid_idx]))
                grid_idx += 1
            else:
                expanded.append(token_int)
        expanded_rows.append(expanded)

    max_len = max((len(row) for row in expanded_rows), default=1)
    out_ids = np.full((len(expanded_rows), max_len), pad_token_id, dtype=input_ids.dtype)
    out_mask = np.zeros((len(expanded_rows), max_len), dtype=attention_mask.dtype)
    for batch_idx, row in enumerate(expanded_rows):
        if not row:
            continue
        row_arr = np.asarray(row, dtype=input_ids.dtype)
        out_ids[batch_idx, : row_arr.size] = row_arr
        out_mask[batch_idx, : row_arr.size] = 1

    return out_ids, out_mask, _build_token_type_ids(config, out_ids, out_mask)


def build_batched_images(
    config: MolmoAct2InputConfig,
    input_ids: np.ndarray,
    pixel_values: np.ndarray,
    image_token_pooling: np.ndarray,
    image_grids: np.ndarray,
    image_num_crops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Regroup per-image crops/pooling into per-example padded tensors.

    Mirrors the PyTorch host-side reassembly: infers the image-to-example
    mapping from ``image_end`` tokens and offsets pooling indices into each
    example's stacked crop patches.

    Returns:
        ``(images, token_pooling)`` of shapes ``(N, max_crops, n_patches, pixels)``
        and ``(N, max_pooled, pool_area)``.
    """
    counts = (input_ids == int(config.image_end_token_id)).sum(1)  # images per example
    num_images = int(image_grids.shape[0])
    if int(counts.sum()) != num_images:
        msg = f"image_end tokens ({int(counts.sum())}) do not match image grids ({num_images})."
        raise ValueError(msg)

    num_examples = counts.shape[0]
    n_crops, n_patches, pixels_per_patch = pixel_values.shape
    del n_crops

    grids = np.asarray(image_grids)
    pooled_per_image = (grids[:, 0] * grids[:, 1] + grids[:, 2] * grids[:, 3]).astype(np.int64)
    example_for_image = np.repeat(np.arange(num_examples), counts)
    crops_per_example = np.zeros(num_examples, dtype=np.int64)
    np.add.at(crops_per_example, example_for_image, image_num_crops.astype(np.int64))
    pooled_per_example = np.zeros(num_examples, dtype=np.int64)
    np.add.at(pooled_per_example, example_for_image, pooled_per_image)
    patches_per_image = image_num_crops.astype(np.int64) * n_patches

    max_crops = int(crops_per_example.max()) if num_examples > 0 else 0
    images = np.full(
        (num_examples, max_crops, n_patches, pixels_per_patch),
        -1.0,
        dtype=pixel_values.dtype,
    )
    max_pooled = int(pooled_per_example.max()) if num_examples > 0 else 0
    token_pooling = np.full(
        (num_examples, max_pooled, image_token_pooling.shape[-1]),
        -1,
        dtype=image_token_pooling.dtype,
    )

    crop_offset = 0
    pooled_offset = 0
    image_offset = 0
    for example_idx in range(num_examples):
        num_example_images = int(counts[example_idx])
        num_example_crops = int(crops_per_example[example_idx])
        images[example_idx, :num_example_crops] = pixel_values[crop_offset : crop_offset + num_example_crops]

        example_pooling = image_token_pooling[
            pooled_offset : pooled_offset + int(pooled_per_example[example_idx])
        ].copy()
        patch_offset = 0
        row = 0
        for local_image in range(num_example_images):
            num_pooled = int(pooled_per_image[image_offset + local_image])
            block = example_pooling[row : row + num_pooled]
            example_pooling[row : row + num_pooled] = np.where(block >= 0, block + patch_offset, block)
            patch_offset += int(patches_per_image[image_offset + local_image])
            row += num_pooled
        token_pooling[example_idx, : example_pooling.shape[0]] = example_pooling

        crop_offset += num_example_crops
        pooled_offset += int(pooled_per_example[example_idx])
        image_offset += num_example_images

    return images, token_pooling


def default_action_dim_is_pad(config: MolmoAct2InputConfig, *, batch_size: int) -> np.ndarray:
    """Mark action dimensions beyond the environment action dim as padding."""
    action_dim_is_pad = np.ones((batch_size, int(config.max_action_dim)), dtype=bool)
    if int(config.env_action_dim) > 0:
        action_dim_is_pad[:, : int(config.env_action_dim)] = False
    return action_dim_is_pad


__all__ = [
    "MolmoAct2InputConfig",
    "build_batched_images",
    "default_action_dim_is_pad",
    "expand_image_placeholders",
]
