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

import pytest

from radiologist.etl.split import assign_split

_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_same_filename_always_returns_the_same_split() -> None:
    filename = "patient_001.png"
    results = {assign_split(filename, _RATIOS) for _ in range(10)}
    assert len(results) == 1


def test_returned_split_is_always_one_of_the_ratio_keys() -> None:
    filenames = [f"image_{i:04d}.png" for i in range(100)]
    for fn in filenames:
        result = assign_split(fn, _RATIOS)
        assert result in _RATIOS


def test_filenames_spread_across_splits_within_5_percent_of_configured_ratio() -> None:
    filenames = [f"image_{i:06d}.png" for i in range(1000)]
    counts: dict[str, int] = {k: 0 for k in _RATIOS}
    for fn in filenames:
        counts[assign_split(fn, _RATIOS)] += 1
    total = len(filenames)
    for split, expected_ratio in _RATIOS.items():
        actual_ratio = counts[split] / total
        assert (
            abs(actual_ratio - expected_ratio) < 0.05
        ), f"{split!r}: expected ~{expected_ratio}, got {actual_ratio:.3f}"


def test_raises_value_error_for_ratios_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError):
        assign_split("image.png", {"a": 0.5, "b": 0.3})


def test_raises_value_error_for_ratios_exceeding_one() -> None:
    with pytest.raises(ValueError):
        assign_split("image.png", {"train": 0.7, "val": 0.4, "test": 0.15})
