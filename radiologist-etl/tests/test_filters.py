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


def _base_df(values: list[float], col: str = "feature") -> pd.DataFrame:
    """Build a minimal DataFrame with excluded/exclusion_reason columns."""
    return pd.DataFrame(
        {
            col: values,
            "excluded": [False] * len(values),
            "exclusion_reason": [""] * len(values),
        }
    )


class TestFilterIqr:
    def test_row_count_unchanged_after_filtering(self):
        df = _base_df([1.0, 2.0, 3.0, 100.0])
        original_count = len(df)

        result = filter_iqr(df, ["feature"])

        assert len(result) == original_count

    def test_rows_within_fence_remain_not_excluded(self):
        # IQR of [1,2,3,4]: Q1=1.75, Q3=3.25, IQR=1.5, fence=[−0.5, 5.5]
        df = _base_df([1.0, 2.0, 3.0, 4.0])

        result = filter_iqr(df, ["feature"])

        assert result["excluded"].tolist() == [False, False, False, False]

    def test_rows_outside_fence_are_flagged_excluded(self):
        # fence for [1,2,3,4] = [-0.5, 5.5]; 100.0 is outside
        df = _base_df([1.0, 2.0, 3.0, 100.0])

        result = filter_iqr(df, ["feature"])

        assert result.iloc[3]["excluded"] is True

    def test_exclusion_reason_contains_iqr_column_name(self):
        df = _base_df([1.0, 2.0, 3.0, 100.0])

        result = filter_iqr(df, ["feature"])

        assert "iqr:feature" in result.iloc[3]["exclusion_reason"]

    def test_multiple_outlier_columns_reasons_joined_with_pipe(self):
        df = pd.DataFrame(
            {
                "col_a": [1.0, 2.0, 3.0, 100.0],
                "col_b": [1.0, 2.0, 3.0, 200.0],
                "excluded": [False, False, False, False],
                "exclusion_reason": ["", "", "", ""],
            }
        )

        result = filter_iqr(df, ["col_a", "col_b"])

        reason = result.iloc[3]["exclusion_reason"]
        assert "iqr:col_a" in reason
        assert "iqr:col_b" in reason
        assert "|" in reason

    def test_returns_same_dataframe_object(self):
        df = _base_df([1.0, 2.0, 3.0, 100.0])

        result = filter_iqr(df, ["feature"])

        assert result is df

    def test_custom_factor_changes_fence(self):
        # With factor=3.0, fence widens so 100 may not be an outlier for tight data
        # [1,2,3,4]: Q1=1.75, Q3=3.25, IQR=1.5, fence(3.0)=[-2.75, 7.75]
        # 100 is still outside either way, but 4.5 would be in fence(3.0) not fence(1.5)
        df = _base_df([1.0, 2.0, 3.0, 4.0, 5.5])
        # fence(1.5) for [1,2,3,4,5.5]: Q1=2.0, Q3=4.0, IQR=2.0, fence=[-1.0, 7.0]
        # 5.5 is within fence(1.5) so should not be flagged
        result = filter_iqr(df, ["feature"], factor=1.5)

        assert result.iloc[4]["excluded"] is False

    def test_already_excluded_row_appends_iqr_reason(self):
        df = pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0, 100.0],
                "excluded": [False, False, False, True],
                "exclusion_reason": ["", "", "", "prior_reason"],
            }
        )

        result = filter_iqr(df, ["feature"])

        reason = result.iloc[3]["exclusion_reason"]
        assert "prior_reason" in reason
        assert "iqr:feature" in reason
        assert "|" in reason


class TestFilterLungOutOfFrame:
    def test_sets_excluded_true_where_lung_out_of_frame_is_true(self):
        df = pd.DataFrame(
            {
                "lung_out_of_frame": [True, False, True],
                "excluded": [False, False, False],
                "exclusion_reason": ["", "", ""],
            }
        )

        result = filter_lung_out_of_frame(df)

        assert result.iloc[0]["excluded"] is True
        assert result.iloc[1]["excluded"] is False
        assert result.iloc[2]["excluded"] is True

    def test_sets_exclusion_reason_lung_out_of_frame(self):
        df = pd.DataFrame(
            {
                "lung_out_of_frame": [True],
                "excluded": [False],
                "exclusion_reason": [""],
            }
        )

        result = filter_lung_out_of_frame(df)

        assert result.iloc[0]["exclusion_reason"] == "lung_out_of_frame"

    def test_noop_when_lung_out_of_frame_column_absent(self):
        df = pd.DataFrame(
            {
                "feature": [1.0, 2.0],
                "excluded": [False, False],
                "exclusion_reason": ["", ""],
            }
        )

        result = filter_lung_out_of_frame(df)

        assert result["excluded"].tolist() == [False, False]
        assert result is df

    def test_appends_reason_when_row_already_excluded(self):
        df = pd.DataFrame(
            {
                "lung_out_of_frame": [True],
                "excluded": [True],
                "exclusion_reason": ["iqr:feature"],
            }
        )

        result = filter_lung_out_of_frame(df)

        reason = result.iloc[0]["exclusion_reason"]
        assert "iqr:feature" in reason
        assert "lung_out_of_frame" in reason
        assert "|" in reason

    def test_returns_same_dataframe_object(self):
        df = pd.DataFrame(
            {
                "lung_out_of_frame": [True],
                "excluded": [False],
                "exclusion_reason": [""],
            }
        )

        result = filter_lung_out_of_frame(df)

        assert result is df

    def test_exclusion_additive_iqr_then_lung_out_of_frame(self):
        """Row already flagged by IQR also gets lung_out_of_frame appended."""
        df = pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0, 100.0],
                "lung_out_of_frame": [False, False, False, True],
                "excluded": [False, False, False, False],
                "exclusion_reason": ["", "", "", ""],
            }
        )

        result = filter_iqr(df, ["feature"])
        result = filter_lung_out_of_frame(result)

        reason = result.iloc[3]["exclusion_reason"]
        assert "iqr:feature" in reason
        assert "lung_out_of_frame" in reason
        assert "|" in reason
