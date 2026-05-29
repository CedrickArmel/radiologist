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

from radiologist.etl.stats import HARALICK_PROPERTIES, lung_asymmetry, make_haralick


def _gray50() -> np.ndarray:
    """50x50 uint8 grayscale synthetic image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(50, 50), dtype=np.uint8)


def _const50(value: int = 128) -> np.ndarray:
    """50x50 constant-value uint8 grayscale image."""
    return np.full((50, 50), value, dtype=np.uint8)


# --- HARALICK_PROPERTIES ---


def test_haralick_properties_is_frozenset_with_exactly_9_names() -> None:
    assert isinstance(HARALICK_PROPERTIES, frozenset)
    assert len(HARALICK_PROPERTIES) == 9
    expected = {
        "mean",
        "std",
        "entropy",
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    }
    assert HARALICK_PROPERTIES == expected


# --- make_haralick defaults ---


def test_make_haralick_with_defaults_returns_9_keys_with_finite_values() -> None:
    extractor = make_haralick()
    image = _gray50()
    result = extractor(image, {})
    assert set(result.keys()) == {f"haralick_{f}" for f in HARALICK_PROPERTIES}
    for key, value in result.items():
        assert math.isfinite(value), f"{key!r} is not finite: {value}"


# --- make_haralick with explicit features ---


def test_make_haralick_with_two_features_returns_exactly_2_keys() -> None:
    extractor = make_haralick(features=["contrast", "energy"])
    result = extractor(_gray50(), {})
    assert set(result.keys()) == {"haralick_contrast", "haralick_energy"}


# --- make_haralick distances/angles collapsing ---


def test_make_haralick_multiple_distances_and_angles_on_constant_image_matches_single() -> (
    None
):
    image = _const50()
    single = make_haralick(distances=[1], angles=[0.0])
    multi = make_haralick(distances=[1, 2], angles=[0.0, math.pi / 2])
    result_single = single(image, {})
    result_multi = multi(image, {})
    for key in result_single:
        assert math.isclose(
            result_single[key], result_multi[key], rel_tol=1e-6, abs_tol=1e-9
        ), f"{key}: {result_single[key]} != {result_multi[key]}"


# --- make_haralick unknown feature raises ValueError ---


def test_make_haralick_raises_value_error_for_unknown_feature_name() -> None:
    with pytest.raises(ValueError, match="unknown"):
        make_haralick(features=["contrast", "nonexistent_feature"])


# --- make_haralick RGB input ---


def test_make_haralick_does_not_crash_on_rgb_input_and_returns_finite_values() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    extractor = make_haralick()
    result = extractor(rgb, {})
    for key, value in result.items():
        assert math.isfinite(value), f"{key!r} is not finite: {value}"


# --- make_haralick picklable ---


def test_extractor_returned_by_make_haralick_is_picklable() -> None:
    extractor = make_haralick()
    data = pickle.dumps(extractor)
    loaded = pickle.loads(data)
    result = loaded(_gray50(), {})
    assert len(result) == 9


# --- lung_asymmetry ---


def test_lung_asymmetry_returns_empty_dict_when_mask_is_none() -> None:
    image = _gray50()
    result = lung_asymmetry(image, {}, mask=None)
    assert result == {}


def test_lung_asymmetry_returns_both_keys_with_symmetric_mask() -> None:
    image = _gray50()
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 5:45] = 1
    result = lung_asymmetry(image, {}, mask=mask)
    assert "asymmetry_ratio" in result
    assert "asymmetry_diff" in result
    assert math.isfinite(result["asymmetry_ratio"])
    assert math.isfinite(result["asymmetry_diff"])


def test_lung_asymmetry_ratio_near_one_for_perfectly_symmetric_mask() -> None:
    image = _gray50()
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Perfectly symmetric: same columns on both halves
    mask[10:40, 0:25] = 1
    mask[10:40, 25:50] = 1
    result = lung_asymmetry(image, {}, mask=mask)
    assert math.isclose(result["asymmetry_ratio"], 1.0, rel_tol=1e-6)
    assert math.isclose(result["asymmetry_diff"], 0.0, abs_tol=1e-9)
