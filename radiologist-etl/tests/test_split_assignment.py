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


def test_same_filename_always_receives_same_split_across_independent_calls() -> None:
    filename = "patient_chest_001.png"
    results = {assign_split(filename, _RATIOS) for _ in range(10)}
    assert len(results) == 1


def test_all_assigned_split_names_are_valid_keys_from_configured_ratios() -> None:
    filenames = [f"scan_{i:04d}.png" for i in range(100)]
    for fn in filenames:
        assert assign_split(fn, _RATIOS) in _RATIOS


def test_population_of_1000_filenames_distributes_within_5_percent_of_configured_ratios() -> (
    None
):
    filenames = [f"image_{i:06d}.png" for i in range(1000)]
    counts: dict[str, int] = {k: 0 for k in _RATIOS}
    for fn in filenames:
        counts[assign_split(fn, _RATIOS)] += 1
    total = len(filenames)
    for split, expected in _RATIOS.items():
        actual = counts[split] / total
        assert (
            abs(actual - expected) < 0.05
        ), f"{split!r}: expected ~{expected:.2f}, got {actual:.3f}"


def test_ratios_that_do_not_sum_to_one_raise_value_error_before_any_assignment() -> (
    None
):
    with pytest.raises(ValueError):
        assign_split("image.png", {"train": 0.5, "val": 0.3})
