# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES, STATE, TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.preprocessors import MolmoAct2ModelInputs, MolmoAct2Preprocessor


def _raw_preprocessor(**kwargs) -> MolmoAct2Preprocessor:
    return MolmoAct2Preprocessor(
        image_keys=["top", "wrist"],
        image_size=(28, 28),
        num_state_tokens=4,
        setup_type="tabletop",
        control_mode="joint",
        **kwargs,
    )


def _model_inputs(**kwargs) -> MolmoAct2ModelInputs:
    return MolmoAct2ModelInputs(
        max_action_dim=4,
        action_dim=2,
        bos_token_id=1,
        pad_token_id=0,
        image_placeholder_token_id=99,
        image_start_token_id=10,
        image_end_token_id=12,
        image_patch_id=11,
        image_col_id=13,
        low_res_image_start_token_id=10,
        image_size=(28, 28),
        patch_size=14,
        pooling_size=(2, 2),
        image_token_ids=[10, 11, 12, 13],
        **kwargs,
    )


class TestMolmoAct2Preprocessor:
    def test_builds_prompt_and_packs_ordered_cameras(self) -> None:
        processor = _raw_preprocessor()
        inputs = {
            STATE: np.array([[-1.0, 1.0]], dtype=np.float32),
            TASK: ["Task: Pick up."],
            f"{IMAGES}.wrist": np.full((1, 3, 28, 28), 255, dtype=np.uint8),
            f"{IMAGES}.top": np.zeros((1, 3, 28, 28), dtype=np.uint8),
        }

        result = processor(inputs)

        assert result[IMAGES].shape == (2, 1, 3, 28, 28)
        assert float(result[IMAGES][0].max()) == 0.0
        assert float(result[IMAGES][1].min()) == 1.0
        assert result[TASK][0].startswith("Image 1<|image|>Image 2<|image|>")
        assert "The task is to pick up." in result[TASK][0]
        assert "<state_0><state_3>" in result[TASK][0]

    def test_applies_masked_normalization_and_joint_transform(self) -> None:
        processor = MolmoAct2Preprocessor(
            image_keys=[],
            image_size=(28, 28),
            state_stats={"q01": [0.0, 0.0], "q99": [2.0, 2.0], "mask": [True, False]},
            adapt_to_so101=True,
            joint_signs=[1.0, -1.0],
            joint_offsets=[0.0, 2.0],
        )
        result = processor(
            {
                STATE: np.array([[1.0, 1.0]], dtype=np.float32),
                TASK: "move",
                IMAGES: np.zeros((1, 3, 28, 28), dtype=np.uint8),
            },
        )
        assert "<state_128><state_255>" in result[TASK][0]

    def test_rejects_missing_state(self) -> None:
        with pytest.raises(ValueError, match="state"):
            _raw_preprocessor()({TASK: ["move"], IMAGES: np.zeros((1, 3, 28, 28), dtype=np.uint8)})

    def test_registry_alias_instantiates(self) -> None:
        processor = instantiate_component(ComponentSpec(type="molmoact2", image_keys=[]))
        assert isinstance(processor, MolmoAct2Preprocessor)


class TestMolmoAct2ModelInputs:
    def test_assembles_model_inputs(self) -> None:
        processor = _model_inputs()
        result = processor(
            {
                TOKENIZED_PROMPT: np.array([[99, 5, 99, 0]], dtype=np.int64),
                TOKENIZED_PROMPT_MASK: np.array([[1, 1, 1, 0]], dtype=np.bool_),
                IMAGES: np.zeros((2, 1, 3, 28, 28), dtype=np.float32),
            },
        )

        assert set(result) == {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "images",
            "token_pooling",
            "action_dim_is_pad",
        }
        assert int(result["input_ids"][0, 0]) == 1
        assert result["images"].shape == (1, 2, 4, 588)
        assert result["token_pooling"].shape == (1, 2, 4)
        np.testing.assert_array_equal(result["action_dim_is_pad"], [[False, False, True, True]])
        assert result["token_type_ids"].sum() > 0

    def test_rejects_placeholder_image_mismatch(self) -> None:
        with pytest.raises(ValueError, match="placeholders"):
            _model_inputs()(
                {
                    TOKENIZED_PROMPT: np.array([[99, 5]], dtype=np.int64),
                    TOKENIZED_PROMPT_MASK: np.ones((1, 2), dtype=np.bool_),
                    IMAGES: np.zeros((2, 1, 3, 28, 28), dtype=np.float32),
                },
            )

    def test_registry_alias_instantiates(self) -> None:
        spec = ComponentSpec(
            type="molmoact2_inputs",
            max_action_dim=4,
            action_dim=2,
            bos_token_id=1,
            pad_token_id=0,
            image_placeholder_token_id=99,
            image_start_token_id=10,
            image_end_token_id=12,
            image_patch_id=11,
            image_col_id=13,
            low_res_image_start_token_id=10,
        )
        assert isinstance(instantiate_component(spec), MolmoAct2ModelInputs)
