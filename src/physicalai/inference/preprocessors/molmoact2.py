# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 inference preprocessor.

Builds MolmoAct2 prompts from raw task/state/images, tokenizes text, inserts BOS,
and emits model-facing tensors that are expected outside the exported model graph.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from physicalai.inference.constants import IMAGES, STATE, TASK

from .base import Preprocessor

ACTION_OUTPUT_TOKEN = "<action_output>"
SETUP_START_TOKEN = "<setup_start>"
SETUP_END_TOKEN = "<setup_end>"
CONTROL_START_TOKEN = "<control_start>"
CONTROL_END_TOKEN = "<control_end>"
STATE_START_TOKEN = "<state_start>"
STATE_END_TOKEN = "<state_end>"
STATE_TOKEN_PREFIX = "<state_"
IMAGE_PROMPT = "<|image|>"
_EPS = 1e-8

_TRAILING_PUNCTUATION = ".,!?;:"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    for pattern in _PREFIX_PATTERNS:
        normalized = pattern.sub("", normalized, count=1).strip()
    normalized = normalized.rstrip(_TRAILING_PUNCTUATION).strip()
    return normalized.lower()


def _wrap_setup_text(setup_type: str, add_setup_tokens: bool) -> str:
    if not setup_type:
        return ""
    if not add_setup_tokens:
        return setup_type
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def _wrap_control_text(control_mode: str, add_control_tokens: bool) -> str:
    if not control_mode:
        return ""
    if not add_control_tokens:
        return control_mode
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


def _build_discrete_state_string(state: np.ndarray, num_state_tokens: int) -> str:
    if num_state_tokens <= 0:
        msg = f"num_state_tokens must be > 0, got {num_state_tokens}."
        raise ValueError(msg)
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    token_ids = np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1).reshape(-1)
    return f"{STATE_START_TOKEN}{''.join(f'{STATE_TOKEN_PREFIX}{int(token_id)}>' for token_id in token_ids)}{STATE_END_TOKEN}"


def _build_robot_text(
    *,
    task: str,
    discrete_state_string: str,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup_text = _wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
    control_text = _wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
    state_clause = f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    prompt = (
        f"The task is to {task}. The setup is {setup_text}.{state_clause} "
        f"The expected control mode is {control_text}. Given these, what action should the robot take to complete the task?"
    )
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = IMAGE_PROMPT
    else:
        image_prefix = "".join(f"Image {idx + 1}{IMAGE_PROMPT}" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{ACTION_OUTPUT_TOKEN}"


class MolmoAct2Preprocessor(Preprocessor):
    """Build MolmoAct2 token and image inputs from raw inference observations."""

    def __init__(
        self,
        tokenizer_name_or_path: str,
        *,
        num_state_tokens: int = 256,
        setup_type: str = "",
        control_mode: str = "",
        add_setup_tokens: bool = False,
        add_control_tokens: bool = False,
        state_stats: dict[str, list[float] | np.ndarray] | None = None,
        image_keys: list[str] | None = None,
    ) -> None:
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.num_state_tokens = int(num_state_tokens)
        self.setup_type = str(setup_type or "")
        self.control_mode = str(control_mode or "")
        self.add_setup_tokens = bool(add_setup_tokens)
        self.add_control_tokens = bool(add_control_tokens)
        self.image_keys = list(image_keys or [])
        self._tokenizer: Any = None

        self._state_q01: np.ndarray | None = None
        self._state_q99: np.ndarray | None = None
        self._state_mask: np.ndarray | None = None
        if state_stats is not None:
            q01 = state_stats.get("q01")
            q99 = state_stats.get("q99")
            if q01 is not None and q99 is not None:
                self._state_q01 = np.asarray(q01, dtype=np.float32)
                self._state_q99 = np.asarray(q99, dtype=np.float32)
            mask = state_stats.get("mask")
            if mask is not None:
                self._state_mask = np.asarray(mask, dtype=bool)

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import Qwen2Tokenizer  # noqa: PLC0415

            self._tokenizer = Qwen2Tokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                local_files_only=False,
            )
        return self._tokenizer

    @staticmethod
    def _insert_bos(
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        bos_token_id: int,
        pad_token_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
            attention_mask = attention_mask[None, :]
            squeeze = True
        else:
            squeeze = False

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            out_ids = np.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype)
            out_mask = np.ones((batch_size, 1), dtype=attention_mask.dtype)
            return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

        first_valid = (attention_mask == 1).argmax(axis=-1)
        if np.all(input_ids[np.arange(batch_size), first_valid] == bos_token_id):
            return (input_ids[0], attention_mask[0]) if squeeze else (input_ids, attention_mask)

        out_ids = np.full((batch_size, seq_len + 1), pad_token_id, dtype=input_ids.dtype)
        out_mask = np.zeros((batch_size, seq_len + 1), dtype=attention_mask.dtype)

        src = np.tile(np.arange(seq_len), (batch_size, 1))
        valid = src >= first_valid[:, None]
        tgt = src + 1
        batch_idx = np.tile(np.arange(batch_size)[:, None], (1, seq_len))

        out_ids[batch_idx[valid], tgt[valid]] = input_ids[valid]
        out_mask[batch_idx[valid], tgt[valid]] = 1
        out_ids[np.arange(batch_size), first_valid] = bos_token_id
        out_mask[np.arange(batch_size), first_valid] = 1
        return (out_ids[0], out_mask[0]) if squeeze else (out_ids, out_mask)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if self._state_q01 is None or self._state_q99 is None:
            return np.clip(state, -1.0, 1.0)

        denom = self._state_q99 - self._state_q01
        denom = np.where(denom == 0, _EPS, denom)
        normalized = 2.0 * (state - self._state_q01) / denom - 1.0
        if self._state_mask is not None:
            mask = self._state_mask
            while mask.ndim < normalized.ndim:
                mask = np.expand_dims(mask, axis=0)
            normalized = np.where(mask, normalized, state)
        return np.clip(normalized, -1.0, 1.0)

    def _extract_state(self, inputs: dict[str, Any]) -> np.ndarray:
        raw_state = inputs.get(STATE)
        if raw_state is None:
            raw_state = inputs.get(f"observation.{STATE}")
        if raw_state is None:
            msg = "MolmoAct2 inference preprocessor requires state."
            raise ValueError(msg)

        state = np.asarray(raw_state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        return self._normalize_state(state)

    @staticmethod
    def _extract_tasks(inputs: dict[str, Any], batch_size: int) -> list[str]:
        task_source = inputs.get(TASK)
        if task_source is None:
            task_source = inputs.get(f"observation.{TASK}")

        if task_source is None:
            tasks = [""] * batch_size
        elif isinstance(task_source, str):
            tasks = [task_source] * batch_size
        elif isinstance(task_source, (list, tuple, np.ndarray)):
            tasks = [str(item) for item in list(task_source)]
        else:
            tasks = [str(task_source)]

        if len(tasks) == 1 and batch_size > 1:
            tasks = tasks * batch_size
        if len(tasks) != batch_size:
            msg = f"Expected {batch_size} task strings, got {len(tasks)}."
            raise ValueError(msg)
        return [_normalize_text(task) for task in tasks]

    def _resolve_image_arrays(self, inputs: dict[str, Any]) -> list[np.ndarray]:
        images_value = inputs.get(IMAGES)
        if isinstance(images_value, dict):
            if self.image_keys:
                return [np.asarray(images_value[key]) for key in self.image_keys if key in images_value]
            return [np.asarray(value) for value in images_value.values()]
        if images_value is not None and not isinstance(images_value, (str, bytes)):
            return [np.asarray(images_value)]

        flat_keys: list[str] = []
        if self.image_keys:
            flat_keys = [f"{IMAGES}.{key}" for key in self.image_keys if f"{IMAGES}.{key}" in inputs]
        if not flat_keys:
            flat_keys = [key for key in inputs if str(key).startswith(f"{IMAGES}.") and "is_pad" not in str(key)]
            flat_keys.sort()
        return [np.asarray(inputs[key]) for key in flat_keys]

    @staticmethod
    def _as_bchw_batch(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim == 3:
            if int(arr.shape[0]) != 3:
                msg = f"Expected CHW image tensor with 3 channels, got shape {arr.shape}"
                raise ValueError(msg)
            arr = arr[None, ...]
        if arr.ndim != 4:
            msg = f"Expected BCHW image tensor, got shape {arr.shape}"
            raise ValueError(msg)
        if int(arr.shape[1]) != 3:
            msg = f"Expected BCHW image tensor with 3 channels, got shape {arr.shape}"
            raise ValueError(msg)

        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        return arr

    def _extract_images_by_example(self, inputs: dict[str, Any], batch_size: int) -> list[list[np.ndarray]]:
        arrays = self._resolve_image_arrays(inputs)
        if not arrays:
            msg = "MolmoAct2 inference preprocessor requires at least one image input."
            raise ValueError(msg)

        images_by_example: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
        for arr in arrays:
            bchw = self._as_bchw_batch(arr)
            if int(bchw.shape[0]) != batch_size:
                msg = f"Image batch size mismatch: expected {batch_size}, got {bchw.shape[0]}"
                raise ValueError(msg)
            for idx in range(batch_size):
                images_by_example[idx].append(bchw[idx])
        return images_by_example

    def __call__(self, inputs: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray]:
        inputs_dict = dict(inputs)

        state = self._extract_state(inputs_dict)
        batch_size = int(state.shape[0])
        tasks = self._extract_tasks(inputs_dict, batch_size)
        images_by_example = self._extract_images_by_example(inputs_dict, batch_size)

        prompt_texts: list[str] = []
        flat_images: list[np.ndarray] = []
        for idx in range(batch_size):
            flat_images.extend(images_by_example[idx])
            discrete_state = _build_discrete_state_string(state[idx], self.num_state_tokens)
            prompt_texts.append(
                _build_robot_text(
                    task=tasks[idx],
                    discrete_state_string=discrete_state,
                    setup_type=self.setup_type,
                    control_mode=self.control_mode,
                    add_setup_tokens=self.add_setup_tokens,
                    add_control_tokens=self.add_control_tokens,
                    num_images=len(images_by_example[idx]),
                )
            )

        image_batch = np.stack(flat_images, axis=0).astype(np.float32) if flat_images else np.empty((0, 3, 0, 0), dtype=np.float32)

        text_inputs = self.tokenizer(prompt_texts, padding=True)
        input_ids = np.asarray(text_inputs["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(text_inputs["attention_mask"], dtype=np.int64)

        bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, attention_mask = self._insert_bos(input_ids, attention_mask, int(bos_token_id), int(pad_token_id))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_placeholder_token_id": np.asarray(
                int(self.tokenizer.convert_tokens_to_ids(IMAGE_PROMPT)),
                dtype=np.int64,
            ),
            "images_bchw": image_batch,
        }
