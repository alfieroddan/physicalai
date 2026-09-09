# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.postprocessors import JointFramePostprocessor


def test_postprocessor_transforms_configured_feature() -> None:
    processor = instantiate_component(
        ComponentSpec(
            type="joint_frame_postprocess",
            feature="action",
            signs=[1.0, -1.0],
            offsets=[10.0, 20.0],
        )
    )
    outputs = {
        "action": np.array([[12.0, 17.0, 4.0]], dtype=np.float32),
        "other": np.array([5.0]),
    }

    assert isinstance(processor, JointFramePostprocessor)
    result = processor(outputs)
    np.testing.assert_array_equal(result["action"], [[2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(result["other"], outputs["other"])
    np.testing.assert_array_equal(outputs["action"], [[12.0, 17.0, 4.0]])


def test_postprocessor_rejects_missing_feature() -> None:
    processor = JointFramePostprocessor(feature="action", signs=[1.0], offsets=[0.0])

    with pytest.raises(ValueError, match="expected feature 'action'"):
        processor({})
