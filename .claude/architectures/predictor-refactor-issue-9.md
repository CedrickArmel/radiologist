## ♻️ Cleanup — remove `Predictor` / `pull_model`, prune `__all__`

### Context

With every capability behavioral (#3–#8) and served/driven through the new hierarchy (#7, #8), this issue removes the now-dead legacy surface. All observable behavior already exists in the new classes and modules; this is the breaking removal the user accepted. The legacy `Predictor` facade, the `pull_model` bridge, and the in-place private helpers in `predictor.py` are deleted, and `__all__` is pruned to the final public surface. Requires: #7, #8.

### Scope

- Delete the `Predictor` class, `pull_model`, `_PredictorState`, and the moved private helpers (`_preprocess_image`, `_read_metadata`, `_apply_prior_correction`) from `predictor.py`; remove the file if nothing public remains in it (the result dataclasses now live in `results.py`, preprocessing in `preprocessing.py`).
- Remove `Predictor` and `pull_model` from `__all__` and from `__init__.py` imports; the final `__all__` is: `BasePredictor`, `Classifier`, `Explainer`, `MCDropoutPredictor`, `score_cam`, `mc_dropout_predict`, `Prediction`, `Explanation`, `UncertaintyResult`, `ModelMetadata`, `create_app`.
- Update `test_public_api.py`'s expected `__all__` set and drop the `Predictor`-specific stub assertions (`test_predictor_explain_raises_not_implemented`, `test_predictor_predict_with_uncertainty_raises_when_no_mcd_session`, the `pull_model`/`create_app` `_wandb`/`_fastapi` patch targets pointing at `predictor` module) — replace each with the equivalent behavioral assertion against the new module that now owns the symbol, never delete silently.
- Retire the legacy-`Predictor` tests that #3/#4/#5 already re-expressed against the new classes (`test_predict.py`, `test_score_cam.py`, `test_mc_dropout.py` portions that import `Predictor`) — only after confirming the equivalent behavioral coverage exists against `Classifier`/`Explainer`/`MCDropoutPredictor`.
- **Not in scope**: any new behavior, route, or CLI change (those are #7/#8).

### Acceptance criteria

- [ ] Importing `radiologist.inference` no longer exposes `Predictor` or `pull_model`; `set(pkg.__all__)` equals the final set above and every name in it is importable.
- [ ] `score_cam` and `mc_dropout_predict` remain importable from `radiologist.inference`.
- [ ] No production module references `Predictor` or `pull_model`.
- [ ] All capability, serving, and CLI behavior verified in #2–#8 still passes (no behavioral test changed to accommodate this cleanup — only legacy-symbol tests are replaced by their new-symbol equivalents).
- [ ] mypy clean; pytest green.
