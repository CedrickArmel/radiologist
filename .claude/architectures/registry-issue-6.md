## ♻️ Wire up & remove legacy registry surfaces

### Context

Final integration/cleanup slice. All registry behavior now lives in `radiologist-registry` (#2, #4, #5) and ONNX export in core's `export_onnx` (#3). This slice rewires package metadata and removes the legacy public surfaces, completing the breaking change. See `registry-spec.md`. Requires: #2, #3, #4, #5.

### Scope

- Add `radiologist-registry` to root `pyproject.toml` `[tool.uv.workspace] members` and `[tool.uv.sources]`; add a `registry = ["radiologist-registry"]` extra alias mirroring existing pattern.
- Add `radiologist-registry` (registry extra) to `radiologist-core` deps so `export.py` can import `ExportResult`.
- Delete `core/registry/pull.py` and `core/registry/promote.py`; update `core/registry/__init__.py` and `core/__init__.py` `__all__` to export `export_onnx` only (remove `pull_checkpoint`, `promote_to_registry`).
- Remove `pull_model` from `radiologist-inference/predictor.py` and its public surface.
- Export the registry public API in `radiologist-registry/src/radiologist/registry/__init__.py` `__all__`.
- **Not in scope**: predictor refactor; any new behavior.

### Acceptance criteria

- [ ] `pull_checkpoint`, `promote_to_registry` are no longer importable from `radiologist.core`; `pull_model` is no longer importable from `radiologist.inference`.
- [ ] `export_onnx` is importable from `radiologist.core`; the full `ModelRegistry` surface is importable from `radiologist.registry`.
- [ ] The resolve → export → promote flow works end-to-end through public APIs across the three packages (against mocked W&B boundary).
- [ ] All existing tests pass without modification except those that directly imported the removed legacy functions, which are migrated to the new surfaces.
- [ ] mypy clean; pytest green across all packages.
