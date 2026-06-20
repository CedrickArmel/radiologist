---
name: project-pytest-layout
description: How pytest is wired in the radiologist monorepo — import mode, rootdir, conftest discovery, src-shim pattern
metadata:
  type: project
---

pytest in this monorepo is configured in root `pyproject.toml` `[tool.pytest.ini_options]`:
`addopts = "--import-mode=importlib ..."` and `testpaths` lists all 5 `radiologist-*/tests` dirs; rootdir is the repo root.

**Why:** packages use `namespace = true` (no `radiologist/__init__.py`) and live under `src/radiologist/`; each `tests/conftest.py` historically did `sys.path.insert(0, .../src)` to make `radiologist.*` importable. Tests import canonical helpers via `from conftest import <fn>` (e.g. `test_app.py`) — this works because under importlib mode the package `tests/` dir (conftest location) is importable.

**How to apply:** A repo-root `conftest.py` is collected once for ALL testpaths (session-global) — natural single home for the shim, but it must insert ALL 5 package `src/` dirs, not one. When designing test-infra refactors, preserve the `from conftest import ...` pattern (keep canonical fn names in the package conftest) and remember the `dm`-style composed fixtures must be function-scoped if they transitively depend on `tmp_path`.

**State as of 2026-06-20 (verify before relying):** root `pyproject.toml` ALREADY has `[tool.pytest.ini_options]` with `testpaths` (all 5 dirs) and `addopts = "--import-mode=importlib --cov=radiologist --cov-report=term-missing"`. No root `conftest.py` exists yet — the shim still lives in each leaf `tests/conftest.py`. `WandbRegistry` is ALREADY exported from `radiologist.registry.__init__`. The inference conftest ALREADY defines `build_det_onnx(tmp_path, priors=None, filename=...)`, `build_mcd_onnx`, `_add_metadata`, `det_onnx_path`, `mcd_onnx_path` — but NOT `det_onnx_path_nonzero`, `predictor_with_mcd`, `predictor_without_mcd`, `sample_image`, nor a `feat_nonzero` param. `test_readers.py` already uses the real-`read_image` + mocked-`fsspec.url_to_fs` pattern (with a `_png_bytes()` helper) in its `test_read_image_remote_*` tests.
