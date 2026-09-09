# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.joint_transform import JointFrameTransform
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.preprocessors import JointFramePreprocessor


def test_joint_transform_round_trip_uses_supplied_frame() -> None:
    transform = JointFrameTransform(signs=(1.0, -1.0), offsets=(10.0, 20.0))
    robot_values = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)

    checkpoint_values = transform.to_checkpoint(robot_values)

    np.testing.assert_array_equal(checkpoint_values, [[12.0, 17.0, 4.0]])
    np.testing.assert_array_equal(transform.to_robot(checkpoint_values), robot_values)


def test_joint_transform_rejects_invalid_frame() -> None:
    with pytest.raises(ValueError, match="must match"):
        JointFrameTransform(signs=(1.0,), offsets=(0.0, 1.0))
    with pytest.raises(ValueError, match="either -1 or 1"):
        JointFrameTransform(signs=(2.0,), offsets=(0.0,))


def test_preprocessor_transforms_configured_feature() -> None:
    processor = instantiate_component(
        ComponentSpec(
            type="joint_frame_preprocess",
            feature="state",
            signs=[1.0, -1.0],
            offsets=[10.0, 20.0],
        )
    )
    inputs = {
        "state": np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        "other": np.array([5.0]),
    }

    assert isinstance(processor, JointFramePreprocessor)
    result = processor(inputs)
    np.testing.assert_array_equal(result["state"], [[12.0, 17.0, 4.0]])
    np.testing.assert_array_equal(result["other"], inputs["other"])
    np.testing.assert_array_equal(inputs["state"], [[2.0, 3.0, 4.0]])


def test_preprocessor_accepts_observation_prefixed_feature() -> None:
    processor = JointFramePreprocessor(feature="state", signs=[-1.0], offsets=[2.0])

    result = processor({"observation.state": np.array([[3.0]], dtype=np.float32)})

    np.testing.assert_array_equal(result["observation.state"], [[-1.0]])


def test_preprocessor_rejects_missing_feature() -> None:
    processor = JointFramePreprocessor(feature="state", signs=[1.0], offsets=[0.0])

    with pytest.raises(ValueError, match="expected feature 'state'"):
        processor({})
