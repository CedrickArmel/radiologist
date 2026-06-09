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

from __future__ import annotations

import numpy as np
import pandas as pd


def _flag_excluded(df: pd.DataFrame, mask: pd.Series, reason_code: str) -> None:
    """Mark rows selected by mask as excluded with the given reason code.

    Args:
        df: DataFrame with `excluded` bool column and `exclusion_reason` str column.
        mask: boolean Series selecting rows to flag.
        reason_code: reason string to append (pipe-separated if already set).
    """
    sep = df.loc[mask, "exclusion_reason"].ne("")
    df.loc[mask, "exclusion_reason"] = np.where(
        sep,
        df.loc[mask, "exclusion_reason"] + "|" + reason_code,
        reason_code,
    )
    df.loc[mask, "excluded"] = True


def filter_iqr(
    df: pd.DataFrame,
    columns: list[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Flag rows whose values fall outside the IQR fence on any specified column.

    Args:
        df: DataFrame with at least the specified columns and an `excluded` bool column
            and `exclusion_reason` str column.
        columns: column names to test.
        factor: IQR multiplier for the fence. Default 1.5.

    Returns:
        The input DataFrame with `excluded` and `exclusion_reason` updated in place
        (returned for chaining). Never drops rows.
    """
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        _flag_excluded(df, mask, "iqr:" + col)

    return df


def filter_lung_out_of_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where lung_out_of_frame is True.

    Args:
        df: DataFrame; may or may not have a `lung_out_of_frame` column.

    Returns:
        The input DataFrame with `excluded` and `exclusion_reason` updated.
        No-op if `lung_out_of_frame` column is absent.
    """
    if "lung_out_of_frame" not in df.columns:
        return df

    mask = df["lung_out_of_frame"].eq(True)
    _flag_excluded(df, mask, "lung_out_of_frame")

    return df
