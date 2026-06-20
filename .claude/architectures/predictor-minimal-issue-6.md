## ♻️ Cleanup — prune dead `Predictor` paths and refresh docs/tests structure

### Context

After #2–#5 reach GREEN-real, the old monolithic `Predictor`, its optional-second-session machinery, and the removed `pull_model` bridge are fully superseded. All observable behavior is implemented and tested through the new subclasses. This issue removes the now-dead code and aligns the test layout and README with the new capability-per-class surface, changing no behavior. Minimal-impact deferred all of this here so the slices stayed small.

> Requires: #2, #3, #4, #5.
> Blocks: —

### Scope

- Delete the residual `Predictor` class, the `pull_model` function, and any `det_session`/`mcd_session` remnants left in `predictor.py` once nothing references them.
- Rename/repoint the structurally-coupled tests that referenced removed symbols (`Predictor`, `pull_model`, `_PredictorState.det_session`) onto the new public surface, naming files by capability (e.g. `test_classification.py`, `test_explanation.py`, `test_uncertainty.py`, `test_serving.py`) rather than by old class.
- Update `radiologist-inference/README.md` quick-start and Public API reference to the new subclasses and the three CLI subcommands.
- **Not in scope**: new behaviors, new routes, bug fixes (open separate Feature/Bug issues).

### Acceptance criteria

- [ ] All existing behavioral tests pass without modification to their assertions (only imports/fixture wiring onto the new public surface may change; if an assertion must change, the slice scope was wrong — fix it there).
- [ ] No symbol named `Predictor` or `pull_model` remains in the source tree.
- [ ] mypy clean; pytest green.
