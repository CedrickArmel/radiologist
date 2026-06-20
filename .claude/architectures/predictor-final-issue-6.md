## ♻️ Cleanup — delete `predictor.py`, finalize public API

### Context

Every behavior the monolith provided now lives in its own module with real
implementations and behavioral tests (#2–#5). This issue removes the dead monolith and
makes the new public surface authoritative. All observable behavior is already implemented
and tested — this is a structural cleanup that must not change any behavior.

**Blocked by:** #2, #3, #4, #5 — the monolith can only be removed once `Classifier`,
`Explainer`, `MCDropoutPredictor`, the smart `create_app`, and the new CLI fully replace
it and the suite is green.

### Scope

- Delete `predictor.py` entirely.
- Confirm `__init__.py` re-exports only the post-epic `__all__` (set in #1) and imports no
  symbol from `predictor`.
- Remove every remaining import of `radiologist.inference.predictor`,
  `Predictor`, `pull_model`, and `_PredictorState` from production and test code.
- Update `test_public_api.py` so no assertion references the removed monolith — drop the
  legacy `Predictor.explain` / `predict_with_uncertainty` / `pull_model` / `create_app`
  cases that patched `radiologist.inference.predictor`; the equivalent behaviors are
  covered through the new class APIs by #2–#5.
- Update `test_registry.py`: the `from_registry` cases now live with `Classifier` (#2);
  remove the `pull_model` cases (that helper is gone — pulling is owned by
  `radiologist-registry`). Delete the file if nothing inference-specific remains.
- **Not in scope:** new behaviors, signature changes, bug fixes (open separate issues).

### Acceptance criteria

- [ ] `predictor.py` no longer exists and no module or test imports from it; `Predictor` and `pull_model` are not importable from `radiologist.inference`.
- [ ] `import radiologist.inference` exposes exactly the post-epic `__all__`.
- [ ] All behavioral tests for prediction, explanation, uncertainty, serving, and CLI pass unchanged from #2–#5 (no test is modified to accommodate the deletion — only legacy-monolith assertions are removed).
- [ ] mypy clean; pytest green.
