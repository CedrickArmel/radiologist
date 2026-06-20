# 🚀 Epic — Decompose `Predictor` into capability subclasses (minimal-impact)

## Problem Statement

The monolithic `Predictor` loads every ONNX model (deterministic + MC-Dropout) and exposes every capability (`predict`, `explain`, `predict_with_uncertainty`) regardless of what the caller needs, so a deployment that only classifies still pays the cost and surface area of explanation and uncertainty — and cannot be containerized or served independently per capability.

## Goal

Replace `Predictor` with a small subclass hierarchy (`BasePredictor` → `Classifier` → `Explainer`, plus `MCDropoutPredictor`) where each subclass loads only the model(s) it needs, is independently servable via a single smart `create_app(predictor)` factory, and is independently containerizable.

## Scope

**In scope:**
- `BasePredictor` (shared load + preprocess + softmax), `Classifier`, `Explainer(Classifier)`, `MCDropoutPredictor`.
- `create_app(predictor)` smart factory that auto-detects the concrete subclass and wires only the matching routes (`/predict`, `/explain`, `/uncertainty`, always `/healthz`).
- Three CLI subcommands: `predict`, `explain`, `uncertainty`. The `pull` subcommand is removed.
- `from_registry()` on each subclass delegating to `WandbRegistry.pull()` (registry epic #85–90).
- Breaking `__all__` change: remove `Predictor` and `pull_model`; add `BasePredictor`, `Classifier`, `Explainer`, `MCDropoutPredictor`.

**Out of scope:**
- Re-implementing Score-CAM / MC-Dropout math — reuse `cam.py` and `mc_dropout_predict` as-is.
- The `WandbRegistry` class itself — owned by registry epic #85–90; this epic only consumes `WandbRegistry.pull()`.
- Splitting `cam.py` or the stateless `score_cam` / `mc_dropout_predict` helpers — they stay public and unchanged.

## Architecture summary

Minimal-impact strategy: **keep `predictor.py` as the single home** for the shared dataclasses (`Prediction`, `Explanation`, `UncertaintyResult`, `ModelMetadata`), the private helpers (`_read_metadata`, `_preprocess_image`, `_apply_prior_correction`), the loaded-state dataclass, and the **entire new class hierarchy**. No new source module is created for the classes; `BasePredictor`, `Classifier`, `Explainer`, and `MCDropoutPredictor` all live in `predictor.py`, reusing the existing helper functions and the existing `cam.py` / `mc_dropout_predict` implementations. `app.py` keeps `_build_app` but it becomes capability-aware: it inspects the injected predictor's type and registers only supported routes. `cli.py` swaps its two commands for three, each constructing the matching subclass. The only behavioral code that moves is route selection (into `_build_app`) and command wiring (in `cli.py`); the inference math is reused verbatim. See the issues files for per-issue contracts.

## Acceptance criteria

- [ ] A `Classifier` loaded from a single deterministic ONNX path can classify an image and is rejected (no route / clear error) for explanation and uncertainty.
- [ ] An `Explainer` can both classify and produce a saliency map from the same single deterministic model.
- [ ] An `MCDropoutPredictor` loaded from a single MC-Dropout ONNX path returns per-class spread and predictive entropy.
- [ ] `create_app(classifier)` serves `/predict` + `/healthz` only; `create_app(explainer)` adds `/explain`; `create_app(mc_dropout_predictor)` serves `/uncertainty` + `/healthz`.
- [ ] CLI exposes `predict`, `explain`, and `uncertainty` subcommands and no `pull` subcommand.
- [ ] Each subclass exposes `from_path` and `from_registry`; `from_registry` delegates to `WandbRegistry.pull()`.
- [ ] `Predictor` and `pull_model` are no longer importable from the package; `BasePredictor`, `Classifier`, `Explainer`, `MCDropoutPredictor` are.
- [ ] mypy clean; pytest green.

## Dependencies

- **Registry epic #85–90** must be merged first — it provides `WandbRegistry.pull()`, which `from_registry()` delegates to, and it removes the `pull_model` bridge. This whole epic sequences after it.
- External packages unchanged: `onnxruntime`, `numpy`, `Pillow` (hard); `wandb` (registry extra), `fastapi`/`uvicorn` (serve extra), `typer` (cli extra).

## Epic shape

1 skeleton issue + 4 outside-in slice issues + 1 optional cleanup issue. See the build sequence table in the parallel-exploration summary and the per-issue files `predictor-decomposition-issue-1.md` … `predictor-decomposition-issue-6.md`.
