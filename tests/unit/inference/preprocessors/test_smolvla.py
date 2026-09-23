# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGE_MASKS, IMAGES
from physicalai.inference.preprocessors import ImageLayout, Preprocessor, ResizeSmolVLA


class TestResizeSmolVLAInit:
    def test_is_preprocessor(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW")
        assert isinstance(prep, Preprocessor)

    def test_default_resolution(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW")
        assert prep.image_resolution == (512, 512)

    def test_custom_resolution(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(256, 256))
        assert prep.image_resolution == (256, 256)


class TestResizeSmolVLACall:
    def test_output_keys(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert IMAGES in result
        assert IMAGE_MASKS in result

    def test_output_shape(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        result = prep({IMAGES: img})
        # 1 image → stacked with extra dim: (1, batch, channels, H, W)
        assert result[IMAGES].shape[2] == 3
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_pixel_range_normalised(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        # input 1.0 → 1.0 * 2 - 1 = 1.0
        np.testing.assert_allclose(result[IMAGES].max(), 1.0, atol=1e-5)

    def test_pixel_range_zeros(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.zeros((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        # input 0.0 → 0.0 * 2 - 1 = -1.0
        np.testing.assert_allclose(result[IMAGES].min(), -1.0, atol=1e-5)

    def test_masks_are_boolean_ones(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.random.rand(2, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGE_MASKS].dtype == np.bool_
        assert result[IMAGE_MASKS].all()

    def test_preserves_other_keys(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img, "task": "pick up"})
        assert result["task"] == "pick up"

    def test_multiple_image_keys(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        inputs = {
            f"{IMAGES}.0": np.random.rand(1, 3, 64, 64).astype(np.float32),
            f"{IMAGES}.1": np.random.rand(1, 3, 64, 64).astype(np.float32),
        }
        result = prep(inputs)
        # Two images stacked
        assert result[IMAGES].shape[0] == 2

    def test_non_square_image_padded(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        # Wide image: height < width
        img = np.random.rand(1, 3, 32, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_non_square_resolution_not_transposed(self) -> None:
        # Regression: image_resolution is (height, width); _resize_with_pad takes (width, height).
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(120, 240))
        img = np.random.rand(1, 3, 60, 60).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 1, 3, 120, 240)

    def test_dict_images_stacked(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        inputs = {
            IMAGES: {
                "top": np.random.rand(1, 3, 48, 48).astype(np.float32),
                "wrist": np.random.rand(1, 3, 32, 64).astype(np.float32),
            },
        }
        result = prep(inputs)
        assert result[IMAGES].shape == (2, 1, 3, 64, 64)
        assert result[IMAGE_MASKS].shape == (2, 1)
        assert result[IMAGE_MASKS].all()


class TestResizeSmolVLADtypeAndLayout:
    def test_uint8_input_normalised(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        # uint8 255 → 1.0 → 1.0 * 2 - 1 = 1.0
        img = np.full((1, 3, 64, 64), 255, dtype=np.uint8)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32
        assert result[IMAGES].shape[1:] == img.shape
        np.testing.assert_allclose(result[IMAGES].max(), 1.0, atol=1e-5)

    def test_uint8_zeros_normalised(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        # uint8 0 → 0.0 → 0.0 * 2 - 1 = -1.0
        img = np.zeros((1, 3, 64, 64), dtype=np.uint8)
        result = prep({IMAGES: img})
        assert result[IMAGES].dtype == np.float32
        assert result[IMAGES].shape[1:] == img.shape
        np.testing.assert_allclose(result[IMAGES].min(), -1.0, atol=1e-5)

    def test_non_float32_dtype_converted(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.float64)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[1:] == img.shape
        assert result[IMAGES].dtype == np.float32

    def test_unsupported_dtype_raises(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.int32)
        with pytest.raises(ValueError, match="Unsupported image dtype"):
            prep({IMAGES: img})

    def test_out_of_range_float_clamped_to_pixel_bounds(self) -> None:
        # Regression: arbitrary float values previously caused output far outside [-1, 1].
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(32, 32))
        img = np.full((1, 3, 32, 32), -1e30, dtype=np.float32)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_nan_float_input_clamped_to_pixel_bounds(self) -> None:
        # Regression: NaN propagates through np.clip unchanged; nan_to_num handles it.
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(32, 32))
        img = np.full((1, 3, 32, 32), float("nan"), dtype=np.float32)
        result = prep({IMAGES: img})
        out = result[IMAGES]
        assert not np.any(np.isnan(out))
        assert float(out.min()) >= -1.0 - 1e-5
        assert float(out.max()) <= 1.0 + 1e-5

    def test_channels_last_transposed(self) -> None:
        prep = ResizeSmolVLA(image_layout="BHWC", image_resolution=(64, 64))
        # (B, H, W, C) input should be transposed to (B, C, H, W) internally
        img = np.random.rand(1, 32, 32, 3).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[2] == 3
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_channels_last_uint8_matches_channels_first(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        chw = np.random.randint(0, 256, size=(1, 3, 48, 48), dtype=np.uint8)
        hwc = np.transpose(chw, (0, 2, 3, 1))
        result_chw = prep({IMAGES: chw})
        prep_hwc = ResizeSmolVLA(image_layout="BHWC", image_resolution=(64, 64))
        result_hwc = prep_hwc({IMAGES: hwc})
        np.testing.assert_array_equal(result_chw[IMAGES], result_hwc[IMAGES])

    @pytest.mark.parametrize("image_layout", ["BCHW", "BHWC"])
    @pytest.mark.parametrize("channels", [1, 2, 3, 4])
    def test_explicit_layout_accepts_supported_channel_counts(self, image_layout: str, channels: int) -> None:
        color = np.arange(1, channels + 1, dtype=np.float32) / channels
        chw = np.broadcast_to(color[None, :, None, None], (1, channels, 8, 10)).copy()
        img = chw if image_layout == "BCHW" else chw.transpose(0, 2, 3, 1)
        prep = ResizeSmolVLA(image_resolution=(16, 20), image_layout=image_layout)

        result = prep({IMAGES: img})

        expected = np.broadcast_to((color * 2 - 1)[None, None, :, None, None], (1, 1, channels, 16, 20))
        np.testing.assert_allclose(result[IMAGES], expected, atol=1e-6)
        np.testing.assert_array_equal(result[IMAGE_MASKS], [[True]])

    @pytest.mark.parametrize("image_layout", ["BCHW", "BHWC"])
    @pytest.mark.parametrize("channels", [0, 5])
    def test_explicit_layout_rejects_unsupported_channel_counts(self, image_layout: str, channels: int) -> None:
        chw = np.zeros((1, channels, 8, 10), dtype=np.float32)
        img = chw if image_layout == "BCHW" else chw.transpose(0, 2, 3, 1)
        prep = ResizeSmolVLA(image_resolution=(16, 20), image_layout=image_layout)

        with pytest.raises(ValueError, match=f"Expected 1 to 4 channels for image_layout={image_layout}") as error:
            prep({IMAGES: img})
        assert f"got {channels} channels" in str(error.value)
        assert str(img.shape) in str(error.value)

    @pytest.mark.parametrize("image_layout", ["BCHW", "BHWC"])
    def test_incorrect_explicit_layout_rejects_spatial_dimension_as_channels(self, image_layout: str) -> None:
        chw = np.zeros((1, 3, 8, 8), dtype=np.float32)
        img = chw.transpose(0, 2, 3, 1) if image_layout == "BCHW" else chw
        prep = ResizeSmolVLA(image_resolution=(16, 16), image_layout=image_layout)

        with pytest.raises(ValueError, match="got 8 channels"):
            prep({IMAGES: img})

    @pytest.mark.parametrize("explicit_none", [False, True])
    def test_automatic_layout_preserves_five_channel_input(self, explicit_none: bool) -> None:
        color = np.arange(1, 6, dtype=np.float32) / 5
        img = np.broadcast_to(color[None, :, None, None], (1, 5, 8, 10)).copy()
        if explicit_none:
            prep = ResizeSmolVLA(image_resolution=(16, 20), image_layout=None)
        else:
            prep = ResizeSmolVLA(image_resolution=(16, 20))

        result = prep({IMAGES: img})
        expected = np.broadcast_to((color * 2 - 1)[None, None, :, None, None], (1, 1, 5, 16, 20))
        np.testing.assert_allclose(result[IMAGES], expected, atol=1e-6)
        np.testing.assert_array_equal(result[IMAGE_MASKS], [[True]])

    @pytest.mark.parametrize("image_layout", [ImageLayout.BCHW, ImageLayout.BHWC, "BCHW", "BHWC"])
    def test_explicit_layout_preserves_pixels_for_ambiguous_shape(self, image_layout: ImageLayout | str) -> None:
        prep = ResizeSmolVLA(image_resolution=(3, 3), image_layout=image_layout)
        img = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3) / 26
        expected = img if image_layout == "BCHW" else img.transpose(0, 3, 1, 2)
        result = prep({IMAGES: img})
        np.testing.assert_array_equal(result[IMAGES], (expected * 2 - 1)[None])
        np.testing.assert_array_equal(result[IMAGE_MASKS], [[True]])

    @pytest.mark.parametrize("image_layout", ["HWC", "CHW", "AUTO", "", "BHWW"])
    def test_invalid_layout_raises(self, image_layout: str) -> None:
        with pytest.raises(ValueError):
            ResizeSmolVLA(image_layout=image_layout)

    @pytest.mark.parametrize("explicit_none", [False, True])
    @pytest.mark.parametrize("image_layout", [ImageLayout.BCHW, ImageLayout.BHWC])
    def test_automatic_layout_matches_explicit(self, explicit_none: bool, image_layout: ImageLayout) -> None:
        chw = np.arange(3 * 7 * 8, dtype=np.float32).reshape(1, 3, 7, 8) / 167
        img = chw if image_layout == ImageLayout.BCHW else chw.transpose(0, 2, 3, 1)
        if explicit_none:
            automatic = ResizeSmolVLA(image_resolution=(12, 16), num_cameras=2, image_layout=None)
        else:
            automatic = ResizeSmolVLA(image_resolution=(12, 16), num_cameras=2)
        explicit = ResizeSmolVLA(image_resolution=(12, 16), num_cameras=2, image_layout=image_layout)

        actual = automatic({IMAGES: img})
        expected = explicit({IMAGES: img})
        np.testing.assert_array_equal(actual[IMAGES], expected[IMAGES])
        np.testing.assert_array_equal(actual[IMAGE_MASKS], expected[IMAGE_MASKS])

    @pytest.mark.parametrize("explicit_none", [False, True])
    def test_automatic_layout_rejects_ambiguous_shape(self, explicit_none: bool) -> None:
        if explicit_none:
            prep = ResizeSmolVLA(image_resolution=(6, 6), image_layout=None)
        else:
            prep = ResizeSmolVLA(image_resolution=(6, 6))
        with pytest.raises(ValueError, match="ambiguous layout"):
            prep({IMAGES: np.zeros((1, 3, 3, 3), dtype=np.float32)})

    def test_automatic_layout_is_inferred_for_each_call(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(7, 8))
        chw = np.arange(3 * 7 * 8, dtype=np.float32).reshape(1, 3, 7, 8) / 167

        for img in (chw, chw.transpose(0, 2, 3, 1), chw):
            result = prep({IMAGES: img})
            np.testing.assert_array_equal(result[IMAGES], (chw * 2 - 1)[None])
            np.testing.assert_array_equal(result[IMAGE_MASKS], [[True]])

    @pytest.mark.parametrize("image_layout", ["BCHW", "BHWC"])
    def test_invalid_ndim_raises_before_transpose(self, image_layout: str) -> None:
        prep = ResizeSmolVLA(image_layout=image_layout)
        with pytest.raises(ValueError, match="4D image"):
            prep({IMAGES: np.zeros((3, 8, 8), dtype=np.float32)})

    @pytest.mark.parametrize("presentation", ["single", "nested", "flat"])
    def test_layouts_produce_identical_images_and_masks(self, presentation: str) -> None:
        chw = np.arange(2 * 3 * 7 * 4, dtype=np.float32).reshape(2, 3, 7, 4) / 167
        outputs = []
        for layout, img in [("BCHW", chw), ("BHWC", chw.transpose(0, 2, 3, 1))]:
            prep = ResizeSmolVLA(image_layout=layout, image_resolution=(8, 12), num_cameras=3)
            if presentation == "nested":
                inputs = {IMAGES: {"top": img, "wrist": img.copy()}}
                # Nested camera inputs are supported at runtime but absent from the input annotation.
                outputs.append(prep(inputs))  # pyrefly: ignore [bad-argument-type]
            elif presentation == "flat":
                outputs.append(prep({"images.top": img, "images.wrist": img.copy()}))
            else:
                outputs.append(prep({IMAGES: img}))
        assert outputs[0][IMAGES].shape == (3, 2, 3, 8, 12)
        np.testing.assert_array_equal(outputs[0][IMAGES], outputs[1][IMAGES])
        np.testing.assert_array_equal(outputs[0][IMAGE_MASKS], outputs[1][IMAGE_MASKS])
        np.testing.assert_array_equal(outputs[0][IMAGE_MASKS][-1], [False, False])
        assert np.all(outputs[0][IMAGES][-1] == -1)

    def test_extreme_aspect_ratio_does_not_crash(self) -> None:
        # Regression: int(cur_height / ratio) could be 0 for extreme ratios, crashing cv2.
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(512, 512))
        img = np.zeros((1, 3, 1, 256), dtype=np.float32)  # very thin image
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[2] >= 1
        assert result[IMAGES].shape[3] >= 1


class TestResizeSmolVLAResizeWithPad:
    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            ResizeSmolVLA._resize_with_pad(np.zeros((3, 64, 64)), 64, 64)

    def test_no_pad_when_matching(self) -> None:
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = ResizeSmolVLA._resize_with_pad(img, 64, 64)
        assert result.shape == (1, 3, 64, 64)

    def test_pads_to_target_size(self) -> None:
        img = np.random.rand(1, 3, 32, 64).astype(np.float32)
        result = ResizeSmolVLA._resize_with_pad(img, 64, 64)
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_zero_height_raises(self) -> None:
        # Regression for fuzzer crash: previously raised ZeroDivisionError.
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(512, 512))
        img = np.zeros((1, 3, 0, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="zero spatial dimension"):
            prep({IMAGES: img})

    def test_zero_width_raises(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(512, 512))
        img = np.zeros((1, 3, 64, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="zero spatial dimension"):
            prep({IMAGES: img})


class TestResizeSmolVLACameraSlots:
    @staticmethod
    def _inputs() -> dict[str, np.ndarray]:
        return {
            f"{IMAGES}.top": np.zeros((1, 3, 64, 64), dtype=np.float32),
            f"{IMAGES}.wrist": np.ones((1, 3, 64, 64), dtype=np.float32),
        }

    def test_reorder_map_orders_cameras(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": 1, "wrist": 0})
        result = prep(self._inputs())
        # wrist (all ones -> +1) occupies slot 0, top (all zeros -> -1) slot 1
        np.testing.assert_allclose(result[IMAGES][0].max(), 1.0, atol=1e-5)
        np.testing.assert_allclose(result[IMAGES][1].max(), -1.0, atol=1e-5)

    def test_reorder_map_accepts_prefixed_keys(self) -> None:
        prep = ResizeSmolVLA(
            image_layout="BCHW",
            image_resolution=(64, 64),
            image_key_reorder_map={f"{IMAGES}.top": 1, f"{IMAGES}.wrist": 0},
        )
        result = prep(self._inputs())
        np.testing.assert_allclose(result[IMAGES][0].max(), 1.0, atol=1e-5)

    def test_reorder_map_key_mismatch_raises(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": 0})
        with pytest.raises(ValueError, match="must match the input image keys exactly"):
            prep(self._inputs())

    def test_negative_slot_index_raises(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": -1, "wrist": 0})

    def test_duplicate_slot_index_raises(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": 0, "wrist": 0})

    def test_num_cameras_pads_empty_slots(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), num_cameras=3)
        result = prep(self._inputs())
        assert result[IMAGES].shape == (3, 1, 3, 64, 64)
        assert result[IMAGE_MASKS].shape == (3, 1)
        np.testing.assert_allclose(result[IMAGES][2], -1.0)
        assert not result[IMAGE_MASKS][2].any()
        assert result[IMAGE_MASKS][0].all()

    def test_num_cameras_with_reorder_map_leaves_gap(self) -> None:
        prep = ResizeSmolVLA(
            image_layout="BCHW",
            image_resolution=(64, 64),
            image_key_reorder_map={"top": 0, "wrist": 2},
            num_cameras=3,
        )
        result = prep(self._inputs())
        np.testing.assert_allclose(result[IMAGES][1], -1.0)
        assert not result[IMAGE_MASKS][1].any()
        np.testing.assert_allclose(result[IMAGES][2].max(), 1.0, atol=1e-5)

    def test_num_cameras_too_small_raises(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), num_cameras=1)
        with pytest.raises(ValueError, match="too small for the resolved camera slots"):
            prep(self._inputs())

    def test_dict_images_respect_reorder_map(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": 1, "wrist": 0})
        inputs = {
            IMAGES: {
                "top": np.zeros((1, 3, 64, 64), dtype=np.float32),
                "wrist": np.ones((1, 3, 64, 64), dtype=np.float32),
            },
        }
        result = prep(inputs)
        np.testing.assert_allclose(result[IMAGES][0].max(), 1.0, atol=1e-5)

    def test_no_images_returns_empty_arrays(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        result = prep({"task": "pick up"})
        assert result[IMAGES].size == 0
        assert result[IMAGE_MASKS].size == 0

    def test_no_images_with_num_cameras_returns_dummy_slots(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(48, 64), num_cameras=2)
        result = prep({"task": "pick up"})
        assert result[IMAGES].shape == (2, 1, 3, 48, 64)
        assert result[IMAGE_MASKS].shape == (2, 1)
        np.testing.assert_allclose(result[IMAGES], -1.0)
        assert not result[IMAGE_MASKS].any()

    def test_single_array_input_with_num_cameras(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), num_cameras=2)
        img = np.ones((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (2, 1, 3, 64, 64)
        np.testing.assert_allclose(result[IMAGES][0].max(), 1.0, atol=1e-5)
        np.testing.assert_allclose(result[IMAGES][1], -1.0)
        assert result[IMAGE_MASKS][0].all()
        assert not result[IMAGE_MASKS][1].any()

    def test_single_array_input_uses_single_entry_reorder_map(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": 1}, num_cameras=3)
        result = prep({IMAGES: np.ones((1, 3, 64, 64), dtype=np.float32)})
        assert result[IMAGES].shape == (3, 1, 3, 64, 64)
        np.testing.assert_allclose(result[IMAGES][1].max(), 1.0, atol=1e-5)
        np.testing.assert_allclose(result[IMAGES][0], -1.0)
        np.testing.assert_allclose(result[IMAGES][2], -1.0)
        assert result[IMAGE_MASKS][1].all()
        assert not result[IMAGE_MASKS][0].any()
        assert not result[IMAGE_MASKS][2].any()

    def test_single_array_input_with_multi_entry_reorder_map_raises(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64), image_key_reorder_map={"top": 1, "wrist": 0}, num_cameras=2)
        with pytest.raises(ValueError, match="must match the input image keys exactly"):
            prep({IMAGES: np.ones((1, 3, 64, 64), dtype=np.float32)})

    def test_single_array_input_unaffected_by_defaults(self) -> None:
        prep = ResizeSmolVLA(image_layout="BCHW", image_resolution=(64, 64))
        img = np.random.rand(2, 3, 32, 32).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape == (1, 2, 3, 64, 64)
        assert result[IMAGE_MASKS].shape == (1, 2)
        assert result[IMAGE_MASKS].all()
