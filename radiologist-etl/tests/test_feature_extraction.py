# MIT License
#
# Copyright (c) 2026 @CedrickArmel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from radiologist.etl import lung_asymmetry, lung_out_of_frame, make_haralick
from radiologist.etl.stats import HARALICK_PROPERTIES


def _gray_image(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (50, 50), dtype=np.uint8)


def _rgb_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)


def _constant_image(value: int = 128) -> np.ndarray:
    return np.full((50, 50), value, dtype=np.uint8)


def test_haralick_extractor_returns_one_finite_float_per_requested_feature() -> None:
    extractor = make_haralick(features=["contrast", "energy"])
    result = extractor(_gray_image(), {})
    assert set(result.keys()) == {"haralick_contrast", "haralick_energy"}
    for key, value in result.items():
        assert math.isfinite(value), f"{key!r} is not finite: {value}"


def test_haralick_extractor_with_default_config_produces_values_for_all_standard_features() -> (
    None
):
    extractor = make_haralick()
    result = extractor(_gray_image(), {})
    assert set(result.keys()) == {f"haralick_{f}" for f in HARALICK_PROPERTIES}
    assert len(result) == 9


def test_requesting_unknown_feature_name_raises_value_error_before_any_image_processing() -> (
    None
):
    with pytest.raises(ValueError, match="unknown"):
        make_haralick(features=["contrast", "not_a_real_feature"])


def test_haralick_extractor_produces_finite_values_for_rgb_input() -> None:
    extractor = make_haralick(features=["contrast", "homogeneity"])
    result = extractor(_rgb_image(), {})
    for key, value in result.items():
        assert math.isfinite(value), f"{key!r} is not finite on RGB input: {value}"


def test_haralick_extractor_produces_finite_values_for_grayscale_input() -> None:
    extractor = make_haralick(features=["contrast", "energy"])
    result = extractor(_gray_image(), {})
    for key, value in result.items():
        assert math.isfinite(
            value
        ), f"{key!r} is not finite on grayscale input: {value}"


def test_haralick_extractor_is_picklable_for_process_pool_use() -> None:
    # Safe: data originates in this process from a known functools.partial — not
    # untrusted input. Pickling is required for ProcessPoolExecutor worker dispatch.
    extractor = make_haralick(features=["contrast"])
    loaded = pickle.loads(pickle.dumps(extractor))
    result = loaded(_gray_image(), {})
    assert math.isfinite(result["haralick_contrast"])


def test_haralick_extractor_multiple_distances_and_angles_on_constant_image_match_single() -> (
    None
):
    image = _constant_image()
    single = make_haralick(features=["homogeneity"], distances=[1], angles=[0.0])
    multi = make_haralick(
        features=["homogeneity"], distances=[1, 2], angles=[0.0, math.pi / 2]
    )
    single_result = single(image, {})
    multi_result = multi(image, {})
    assert math.isclose(
        single_result["haralick_homogeneity"],
        multi_result["haralick_homogeneity"],
        rel_tol=1e-6,
        abs_tol=1e-9,
    )


def test_asymmetry_extractor_returns_empty_result_when_no_mask_provided() -> None:
    result = lung_asymmetry(_gray_image(), {}, mask=None)
    assert result == {}


def test_asymmetry_extractor_returns_finite_ratio_at_least_one_for_lopsided_mask() -> (
    None
):
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 0:10] = 255  # heavy left side
    result = lung_asymmetry(_gray_image(), {}, mask=mask)
    assert "asymmetry_ratio" in result
    assert math.isfinite(result["asymmetry_ratio"])
    assert result["asymmetry_ratio"] >= 1.0


def test_perfectly_symmetric_mask_produces_asymmetry_ratio_near_one() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 0:25] = 255
    mask[10:40, 25:50] = 255
    result = lung_asymmetry(_gray_image(), {}, mask=mask)
    assert math.isclose(result["asymmetry_ratio"], 1.0, rel_tol=1e-6)


def test_lung_touching_image_border_is_detected_as_out_of_frame() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 5] = 255  # first row
    assert lung_out_of_frame(mask) is True


def test_lung_contained_entirely_within_interior_is_not_flagged_as_out_of_frame() -> (
    None
):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255
    assert lung_out_of_frame(mask) is False


def test_framing_check_works_the_same_for_rgb_mask_as_for_single_channel_mask() -> None:
    mask_2d = np.zeros((10, 10), dtype=np.uint8)
    mask_2d[0, 4] = 200

    mask_3d = np.zeros((10, 10, 3), dtype=np.uint8)
    mask_3d[0, 4, 1] = 200

    assert lung_out_of_frame(mask_2d) == lung_out_of_frame(mask_3d)
    assert lung_out_of_frame(mask_2d) is True
