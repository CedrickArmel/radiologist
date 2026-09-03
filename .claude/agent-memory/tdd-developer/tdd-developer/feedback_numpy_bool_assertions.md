---
name: feedback-numpy-bool-assertions
description: Use == True/False not `is True/False` when asserting on pandas/numpy booleans from parquet or DataFrame cells
metadata:
  type: feedback
---

Use `== True` and `== False` (value equality) rather than `is True` / `is False` when asserting on values read back from pandas DataFrames or Parquet files.

**Why:** pandas and pyarrow return `np.True_` / `np.False_` (numpy scalars), not Python builtins. `np.True_ is True` evaluates to `False` even though the value is correct, causing spurious test failures.

**How to apply:** In any test that reads from `pd.read_parquet`, `df.iloc[n][col]`, or any pandas cell, always use `== True`, `== False`, or `pd.isna()` — never identity checks with `is`.
