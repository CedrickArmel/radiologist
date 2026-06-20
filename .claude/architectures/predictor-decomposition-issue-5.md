## ♻️ Cleanup — docs and dead-code sweep after decomposition

### Context

All observable behavior is implemented and tested by #2–#4. This optional slice aligns documentation and removes now-dead scaffolding left by the decomposition (e.g. the old `_PredictorState` dataclass and any `Predictor`-era comments), updating `radiologist-inference/README.md` to describe the new class hierarchy, the smart `create_app`, and the 3-command CLI.

### Scope

- Update `radiologist-inference/README.md`: replace `Predictor`/`pull_model` examples with `Classifier`/`Explainer`/`MCDropoutPredictor`, the smart `create_app`, and `predict`/`explain`/`uncertainty` CLI usage.
- Remove dead internal scaffolding (`_PredictorState` and any unreferenced private helpers) and tidy module docstrings.
- **Not in scope**: new behaviors, signature changes, bug fixes (open separate Feature or Bug issues).

### Acceptance criteria

- [ ] All existing tests pass without modification (no test may be changed to accommodate this refactor — if one must change, the scope is wrong).
- [ ] mypy clean; pytest green.
