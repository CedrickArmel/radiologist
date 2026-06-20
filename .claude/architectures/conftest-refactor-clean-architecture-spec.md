# 🚀 Epic — Conftest & fixture refactor (Clean Architecture)

## Problem Statement

Test fixtures and helpers are duplicated across all 5 packages and many inference test files (the `sys.path` shim ×5, four near-identical ONNX builders, an 8-parameter `_make_dm` boilerplate repeated in ~20 tests), making the test suite expensive to change and prone to drift between copies.

## Goal

A single source of truth for every shared test fixture: one root `conftest.py` owns the `sys.path` shim, the inference `conftest.py` owns every ONNX builder and model-path fixture, and `test_datamodule.py` composes a `WebDatasetDataModule` through one `dm` fixture — with no behavioral test changing its assertions.

## Scope

**In scope:**

- Root `conftest.py` carrying the `sys.path` shim for all 5 packages.
- Promote the 4 pure-value core fixtures (`label_map`, `classes`, `batch_size`, merged `transform`) to `session` scope.
- Merge `train_transform`/`eval_transform` into one session-scoped `transform`.
- Canonicalize all ONNX builders in inference `conftest.py`; expose explicit per-variant fixtures.
- Add a composed function-scoped `dm` fixture to core `conftest.py`.
- Delete all local ONNX builder copies and shadowing fixtures from inference test files.
- Formalize the test layout in `pyproject.toml` (`[tool.pytest.ini_options]`).

**Out of scope:**

- Renaming or re-homing behavioral test functions (Test Contravariance: assertions stay).
- Changing any production code under `src/`.
- Touching `radiologist-app` (does not exist on disk).

## Architecture summary

The refactor is organized as **one shared seam (skeleton) + two independent slices**. The seam (#1) establishes every shared fixture surface: the root `conftest.py` shim, the core conftest's session-scoped value fixtures + merged `transform` + composed `dm`, and the inference conftest's canonical builders + explicit variant fixtures (`det_onnx_path`, `det_onnx_path_nonzero`, `mcd_onnx_path`, `predictor_with_mcd`, `predictor_without_mcd`, `sample_image`). Both slices depend only on the seam and can proceed in parallel: Slice A (#2) deletes local builder copies and shadowing fixtures across the inference test files so they consume the canonical conftest fixtures; Slice B (#3) rewrites `test_datamodule.py` to take the single `dm` fixture in the common case. The Clean Architecture stance drives two specific choices documented in the issues: **explicit variant fixtures over boolean-flag builders** (a `det_onnx_path_nonzero` fixture rather than `build_det_onnx(feat_nonzero=True)` at call sites), and **formalized layout config** in `pyproject.toml`. See the issue files for per-file contracts.

## Acceptance criteria

- [ ] The `sys.path` shim is defined exactly once and every package's tests still import from its `src/`.
- [ ] No ONNX-building function is defined in any `test_*.py` file under `radiologist-inference/tests/`.
- [ ] `test_datamodule.py` common-case tests build the datamodule through a single fixture parameter.
- [ ] `label_map`, `classes`, `batch_size`, and the merged `transform` are session-scoped.
- [ ] The full suite (`make test`) passes with no behavioral assertion modified; mypy clean.

## Dependencies

- None external. Self-contained refactor of the existing test suite.
- Build order: #1 (skeleton/seam) blocks #2 and #3.
