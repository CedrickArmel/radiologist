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

"""Split ratios and per-filename split assignment (ordered-sequence contract).

Ratios are an explicitly ordered sequence of ``(name, fraction)`` pairs — the
order is part of the split contract, not a formatting detail. A plain
mapping is rejected outright rather than silently coerced, because coercing
it would restore exactly the hidden order-dependence being removed.
"""

from __future__ import annotations

import hashlib

import pytest

_DEFAULT_RATIOS = [("train", 0.70), ("val", 0.15), ("test", 0.15)]


def _reference_assign(filename: str, ratios: list) -> str:
    """Independent reimplementation of the pre-#184 MD5-bracket algorithm.

    Used to prove the new ordered-sequence implementation reproduces exactly
    what the previous dict-ratios pipeline assigned for the shipped default
    order/values.
    """
    modulus = 16**32
    hex_digest = hashlib.md5(filename.encode()).hexdigest()
    fraction = int(hex_digest, 16) / modulus
    cumulative = 0.0
    for name, ratio in ratios:
        cumulative += ratio
        if fraction < cumulative:
            return name
    return ratios[-1][0]


class TestNormalizeRatios:
    def test_rejects_plain_mapping_because_order_is_part_of_the_contract(
        self,
    ) -> None:
        from radiologist.etl import normalize_ratios

        with pytest.raises(ValueError, match="order"):
            normalize_ratios({"train": 0.70, "val": 0.15, "test": 0.15})

    def test_accepts_ordered_sequence_and_returns_it_as_a_list(self) -> None:
        from radiologist.etl import normalize_ratios

        result = normalize_ratios(_DEFAULT_RATIOS)
        assert result == _DEFAULT_RATIOS

    def test_rejects_ratios_that_do_not_sum_to_one_reporting_observed_sum(
        self,
    ) -> None:
        from radiologist.etl import normalize_ratios

        with pytest.raises(ValueError, match="0.8"):
            normalize_ratios([("train", 0.5), ("val", 0.3)])

    def test_rejects_a_negative_fraction(self) -> None:
        from radiologist.etl import normalize_ratios

        with pytest.raises(ValueError):
            normalize_ratios([("train", 1.1), ("val", -0.1)])

    def test_rejects_a_repeated_split_name(self) -> None:
        from radiologist.etl import normalize_ratios

        with pytest.raises(ValueError):
            normalize_ratios([("train", 0.5), ("train", 0.5)])

    def test_rejects_an_empty_sequence(self) -> None:
        from radiologist.etl import normalize_ratios

        with pytest.raises(ValueError):
            normalize_ratios([])


class TestAssignSplit:
    def test_same_filename_always_receives_same_split_across_independent_calls(
        self,
    ) -> None:
        from radiologist.etl import assign_split

        filename = "patient_chest_001.png"
        results = {assign_split(filename, _DEFAULT_RATIOS) for _ in range(10)}
        assert len(results) == 1

    def test_all_assigned_split_names_are_valid_names_from_configured_ratios(
        self,
    ) -> None:
        from radiologist.etl import assign_split

        filenames = [f"scan_{i:04d}.png" for i in range(100)]
        valid_names = {name for name, _ in _DEFAULT_RATIOS}
        for fn in filenames:
            assert assign_split(fn, _DEFAULT_RATIOS) in valid_names

    def test_population_of_1000_filenames_distributes_within_5_percent_of_configured_ratios(
        self,
    ) -> None:
        from radiologist.etl import assign_split

        filenames = [f"image_{i:06d}.png" for i in range(1000)]
        counts: dict[str, int] = {name: 0 for name, _ in _DEFAULT_RATIOS}
        for fn in filenames:
            counts[assign_split(fn, _DEFAULT_RATIOS)] += 1
        total = len(filenames)
        for split, expected in _DEFAULT_RATIOS:
            actual = counts[split] / total
            assert (
                abs(actual - expected) < 0.05
            ), f"{split!r}: expected ~{expected:.2f}, got {actual:.3f}"

    def test_ratios_that_do_not_sum_to_one_raise_value_error_before_any_assignment(
        self,
    ) -> None:
        from radiologist.etl import assign_split

        with pytest.raises(ValueError):
            assign_split("image.png", [("train", 0.5), ("val", 0.3)])

    def test_a_plain_mapping_is_rejected_because_order_is_part_of_the_contract(
        self,
    ) -> None:
        from radiologist.etl import assign_split

        with pytest.raises(ValueError, match="order"):
            assign_split("image.png", {"train": 0.70, "val": 0.15, "test": 0.15})

    def test_default_ordered_ratios_reproduce_the_previous_pipelines_assignment(
        self,
    ) -> None:
        from radiologist.etl import assign_split

        filenames = [f"patient_{i:05d}.png" for i in range(200)]
        for fn in filenames:
            assert assign_split(fn, _DEFAULT_RATIOS) == _reference_assign(
                fn, _DEFAULT_RATIOS
            )
