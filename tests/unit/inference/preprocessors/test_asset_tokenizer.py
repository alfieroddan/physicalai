# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component, resolve_artifact
from physicalai.inference.constants import TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.preprocessors import AssetTokenizer


def _mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.name_or_path = "local-tokenizer"

    def _encode(tasks, **kwargs):
        length = kwargs["max_length"]
        return {
            "input_ids": np.ones((len(tasks), length), dtype=np.int64),
            "attention_mask": np.ones((len(tasks), length), dtype=np.int64),
        }

    tokenizer.side_effect = _encode
    return tokenizer


def test_loads_allowlisted_tokenizer_with_dynamic_options(tmp_path: Path) -> None:
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")
    transformers = MagicMock()
    tokenizer = _mock_tokenizer()
    transformers.Qwen2Tokenizer.from_pretrained.return_value = tokenizer
    options = {
        "bos_token": "<|im_end|>",
        "extra_special_tokens": ["<im_start>", "<|image|>"],
        "model_max_length": 1010000,
    }

    with patch.dict("sys.modules", {"transformers": transformers}):
        preprocessor = AssetTokenizer(
            artifact=str(tokenizer_file),
            tokenizer_class="Qwen2Tokenizer",
            tokenizer_options=options,
            max_token_len=4,
        )

    transformers.Qwen2Tokenizer.from_pretrained.assert_called_once_with(
        tmp_path,
        local_files_only=True,
        **options,
    )
    result = preprocessor({TASK: ["pick up the block"]})
    assert result[TOKENIZED_PROMPT].shape == (1, 4)
    assert result[TOKENIZED_PROMPT_MASK].dtype == np.bool_
    assert TASK not in result


def test_rejects_unsupported_tokenizer_class(tmp_path: Path) -> None:
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported asset tokenizer class"):
        AssetTokenizer(artifact=str(tokenizer_file), tokenizer_class="ArbitraryTokenizer")


def test_manifest_resolves_flat_artifact(tmp_path: Path) -> None:
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")
    transformers = MagicMock()
    transformers.Qwen2Tokenizer.from_pretrained.return_value = _mock_tokenizer()
    spec = resolve_artifact(
        ComponentSpec(
            type="asset_tokenizer",
            artifact="tokenizer.json",
            tokenizer_class="Qwen2Tokenizer",
            tokenizer_options={"extra_special_tokens": ["<|image|>"]},
        ),
        tmp_path,
    )

    with patch.dict("sys.modules", {"transformers": transformers}):
        component = instantiate_component(spec)

    assert isinstance(component, AssetTokenizer)
    assert spec.flat_params["artifact"] == str(tokenizer_file)