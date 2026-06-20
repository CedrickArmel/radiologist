# 🚀 Epic — `radiologist-registry`: a dedicated W&B model-registry package

## Problem Statement

W&B registry concerns (checkpoint resolution, ONNX export, artifact upload/link, stage-tag management) are scattered and coupled across `radiologist-core` (`pull_checkpoint`, `promote_to_registry` — which fuses ONNX export with W&B upload) and `radiologist-inference` (`pull_model`), making registry behavior impossible to evolve or test in isolation.

## Goal

A single `radiologist-registry` package exposes one minimal `ModelRegistry` Protocol (concrete `WandbRegistry`) that owns every W&B-side operation — resolve, pull, promote (upload+link), and stage-tag management — while ONNX export stays in core and hands over an `ExportResult`.

## Scope

**In scope:**
- New workspace member `radiologist-registry` (mirrors existing package layout/conventions).
- `ModelRegistry` Protocol + `WandbRegistry` concrete implementation.
- `ExportResult` dataclass — the hand-off contract between core's ONNX export and the registry's `promote`.
- Stage-tag (alias) management: `set_alias`, `remove_alias`, `get_aliases`.
- Core refactor: `promote_to_registry` split per **Option B** — core keeps `export_onnx` returning `ExportResult`; all W&B upload/link moves to the registry.
- Breaking removals: `pull_checkpoint` + `promote_to_registry` out of `core.__all__`; `pull_model` out of inference.

**Out of scope:**
- Predictor refactor (separate epic). Inference still consumes ONNX paths the same way; only its internal `pull_model` is removed in favor of `WandbRegistry.pull`.
- Any new resolution semantics beyond what `_resolve_model_artifact` already does.

## Architecture summary

Minimal-surface design: one Protocol (`ModelRegistry`) capturing exactly the W&B boundary the codebase needs — `resolve_checkpoint`, `pull`, `promote`, and the three alias ops — with a single concrete `WandbRegistry`. The existing `_resolve_model_artifact` resolution logic moves verbatim into `WandbRegistry.resolve_checkpoint`/`pull`; the upload+link tail of `promote_to_registry` becomes `WandbRegistry.promote(ExportResult, ...)`. ONNX export (the `_CamWrapper`, dual export, metadata stamping) stays in `radiologist-core` as a pure `export_onnx(...) -> ExportResult` with zero W&B dependency. `ExportResult` is a frozen dataclass living in the registry package (the shared type), imported by core. W&B stays an optional dep guarded by a module-level sentinel, mirroring `inference/optional.py`. See the per-issue files for contracts.

## Acceptance criteria

- [ ] A caller can resolve a checkpoint, pull it locally, and get the same path the old `pull_checkpoint` returned, via `WandbRegistry`.
- [ ] A caller can export ONNX models in core (no wandb installed) and receive an `ExportResult`, then promote that result via `WandbRegistry.promote` to obtain the linked artifact's qualified name.
- [ ] A caller can add, remove, and read stage aliases on an artifact.
- [ ] Importing `radiologist.core` no longer exposes `pull_checkpoint` or `promote_to_registry`; importing `radiologist.inference` no longer exposes `pull_model`.
- [ ] All packages: mypy clean; pytest green.

## Dependencies

- W&B (`wandb`) — optional extra, already a project dependency.
- `onnx`, `torch` — already present in `radiologist-core` for export.
- No external blockers.
