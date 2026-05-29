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

import pandas as pd

from radiologist.etl.filters import filter_iqr, filter_lung_out_of_frame


def _manifest_df(values: list[float], col: str = "haralick_contrast") -> pd.DataFrame:
    """Minimal manifest DataFrame with the required bookkeeping columns."""
    return pd.DataFrame(
        {
            col: values,
            "excluded": [False] * len(values),
            "exclusion_reason": [""] * len(values),
        }
    )


def test_statistical_outliers_are_flagged_but_row_count_is_unchanged() -> None:
    df = _manifest_df([1.0, 2.0, 3.0, 1000.0])
    original_len = len(df)
    result = filter_iqr(df, ["haralick_contrast"])
    assert len(result) == original_len
    assert bool(result.iloc[3]["excluded"]) is True


def test_outlier_exclusion_reason_contains_column_name_with_iqr_prefix() -> None:
    df = _manifest_df([1.0, 2.0, 3.0, 1000.0])
    result = filter_iqr(df, ["haralick_contrast"])
    assert "iqr:haralick_contrast" in result.iloc[3]["exclusion_reason"]


def test_row_that_is_outlier_on_two_columns_accumulates_both_reasons() -> None:
    df = pd.DataFrame(
        {
            "col_a": [1.0, 2.0, 3.0, 999.0],
            "col_b": [1.0, 2.0, 3.0, 888.0],
            "excluded": [False, False, False, False],
            "exclusion_reason": ["", "", "", ""],
        }
    )
    result = filter_iqr(df, ["col_a", "col_b"])
    reason = result.iloc[3]["exclusion_reason"]
    assert "iqr:col_a" in reason
    assert "iqr:col_b" in reason
    assert "|" in reason


def test_images_within_statistical_fence_are_not_flagged() -> None:
    # Q1=1.75, Q3=3.25, IQR=1.5, fence=[-0.5, 5.5] for [1,2,3,4]
    df = _manifest_df([1.0, 2.0, 3.0, 4.0])
    result = filter_iqr(df, ["haralick_contrast"])
    assert result["excluded"].tolist() == [False, False, False, False]


def test_scans_with_lung_touching_border_are_excluded_from_training() -> None:
    df = pd.DataFrame(
        {
            "lung_out_of_frame": [True, False, True],
            "excluded": [False, False, False],
            "exclusion_reason": ["", "", ""],
        }
    )
    result = filter_lung_out_of_frame(df)
    assert bool(result.iloc[0]["excluded"]) is True
    assert bool(result.iloc[1]["excluded"]) is False
    assert bool(result.iloc[2]["excluded"]) is True


def test_framing_filter_is_no_op_when_no_lung_masks_were_computed() -> None:
    df = pd.DataFrame(
        {
            "haralick_contrast": [1.0, 2.0],
            "excluded": [False, False],
            "exclusion_reason": ["", ""],
        }
    )
    result = filter_lung_out_of_frame(df)
    assert result["excluded"].tolist() == [False, False]
    assert result is df


def test_row_already_flagged_by_iqr_also_gets_lung_out_of_frame_reason_appended() -> (
    None
):
    df = pd.DataFrame(
        {
            "haralick_contrast": [1.0, 2.0, 3.0, 999.0],
            "lung_out_of_frame": [False, False, False, True],
            "excluded": [False, False, False, False],
            "exclusion_reason": ["", "", "", ""],
        }
    )
    result = filter_iqr(df, ["haralick_contrast"])
    result = filter_lung_out_of_frame(result)
    reason = result.iloc[3]["exclusion_reason"]
    assert "iqr:haralick_contrast" in reason
    assert "lung_out_of_frame" in reason
    assert "|" in reason
