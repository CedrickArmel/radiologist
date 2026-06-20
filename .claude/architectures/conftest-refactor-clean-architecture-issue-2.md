## ✨ Slice A — Inference test cleanup (consume canonical conftest fixtures)

### Context

The five inference test files each carry their own copy of the ONNX builders (`_build_det_onnx`, `_build_mcd_onnx`, `_add_metadata`) and some define fixtures that shadow the conftest versions. This slice deletes every local copy and rewires each file to consume the canonical builders and fixtures established in the seam (#1). It implements no new behavior — every test keeps its current assertions. This is the contravariant move: the shared surface narrows to conftest, the tests get thinner. Requires: #1. See the epic spec for context.

### User story

As a **maintainer of the inference test suite**, I want **every ONNX model builder defined once in conftest** so that **a change to the model contract is made in exactly one place and cannot drift between test files**.

### Acceptance criteria

<!-- Behavioral assertions are unchanged; these criteria assert the observable end state of the
     cleanup and that the suite still passes. -->

- [ ] Running the inference test suite produces the same pass/fail outcomes as before this slice (no assertion changed).
- [ ] `test_mc_dropout.py` no longer defines its own deterministic or stochastic ONNX builder, and no longer defines `det_onnx_path`, `mcd_onnx_path`, `predictor_with_mcd`, `predictor_without_mcd`, or `sample_image` — these resolve to the conftest fixtures.
- [ ] `test_score_cam.py` obtains its nonzero-feature-map model through the `det_onnx_path_nonzero` fixture (or `build_det_onnx(..., feat_nonzero=True)`), not a locally defined builder, and its Score-CAM assertions still hold (feature maps are nonzero).
- [ ] `test_predict.py` obtains deterministic models (including prior-bearing variants via `build_det_onnx(..., priors=...)`) from conftest, with no local builder.
- [ ] `test_wandb_registry_pull.py` obtains its deterministic model from conftest, with no local builder.
- [ ] `test_app.py`'s module-level `from conftest import build_det_onnx, build_mcd_onnx` still resolves and its endpoint tests pass.
- [ ] mypy clean; pytest green

### Out of scope

- Renaming any test function or class, or changing any assertion.
- The `dm` fixture / `test_datamodule.py` (that is Slice B, #3).

### Technical notes

- `radiologist-inference/tests/test_score_cam.py` — its local builder uses `np.random.rand` for `feat_const` (Score-CAM needs nonzero feature maps) and its `_add_metadata` has no default `extra`. The canonical `build_det_onnx(..., feat_nonzero=True)` reproduces the nonzero behavior; route this file through `det_onnx_path_nonzero`.
- `radiologist-inference/tests/test_predict.py` — its local builder accepts `priors`; preserve prior-bearing cases by calling the canonical `build_det_onnx(tmp_path, priors=...)`.
- `radiologist-inference/tests/test_mc_dropout.py` — currently re-defines `det_onnx_path`/`mcd_onnx_path` (shadowing conftest) plus `predictor_with_mcd`/`predictor_without_mcd`/`sample_image`; delete all of these so the conftest fixtures take over.
- `radiologist-inference/tests/test_app.py` — keep the `from conftest import ...` line; the canonical functions remain module-level in conftest, so the import path is unchanged.
