# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer preprocessor loaded from a bundled tokenizer artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.preprocessors.base import Preprocessor

_SUPPORTED_TOKENIZER_CLASSES = {"Qwen2Tokenizer"}


class AssetTokenizer(Preprocessor):
    """Load an allowlisted Transformers tokenizer from a local artifact.

    Args:
        artifact: Path to a bundled tokenizer file such as ``tokenizer.json``.
        tokenizer_class: Allowlisted Transformers tokenizer class name.
        tokenizer_options: Checkpoint-derived tokenizer construction options.
        max_token_len: Maximum encoded prompt length.
    """

    def __init__(
        self,
        artifact: str,
        tokenizer_class: str,
        tokenizer_options: dict[str, Any] | None = None,
        max_token_len: int = 512,
    ) -> None:
        """Initialize a tokenizer from a bundled local artifact.

        Raises:
            FileNotFoundError: If the tokenizer artifact does not exist.
            ImportError: If Transformers is not installed.
            ValueError: If the requested tokenizer class is not supported.
        """
        super().__init__()
        artifact_path = Path(artifact)
        if not artifact_path.is_file():
            msg = f"Tokenizer artifact does not exist: {artifact_path}"
            raise FileNotFoundError(msg)
        if tokenizer_class not in _SUPPORTED_TOKENIZER_CLASSES:
            msg = f"Unsupported asset tokenizer class: {tokenizer_class!r}"
            raise ValueError(msg)

        try:
            import transformers  # ruff: ignore[PLC0415]
        except ImportError as exc:
            msg = "Tokenizer requires transformers. Install with: pip install transformers"
            raise ImportError(msg) from exc

        tokenizer_type = getattr(transformers, tokenizer_class)
        self._tokenizer = tokenizer_type.from_pretrained(
            artifact_path.parent,
            local_files_only=True,
            **(tokenizer_options or {}),
        )
        self._max_token_len = max_token_len

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Tokenize tasks and replace them with token IDs and masks.

        Returns:
            Input values with tasks replaced by token IDs and attention masks.

        Raises:
            TypeError: If the task value is not a list.
        """
        batch_tasks = inputs[TASK]
        if not isinstance(batch_tasks, list):
            msg = f"Expected TASK to be a list of strings, got {type(batch_tasks)}"
            raise TypeError(msg)

        outputs = dict(inputs)
        outputs.pop(TASK)
        encoded_tokens = self._tokenizer(
            batch_tasks,
            max_length=self._max_token_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        outputs[TOKENIZED_PROMPT] = encoded_tokens["input_ids"]
        outputs[TOKENIZED_PROMPT_MASK] = encoded_tokens["attention_mask"].astype(np.bool_)
        return outputs

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return f"{self.__class__.__name__}(tokenizer={self._tokenizer.name_or_path!r})"
