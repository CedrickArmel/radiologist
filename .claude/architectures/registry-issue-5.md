## ♻️ Remove legacy registry APIs and rewire callers

### Context

With the registry package fully behavioral (#2–#4), this issue removes the now-dead legacy surface and points existing callers at `WandbRegistry`. All observable behavior already exists in the new package; this is the breaking cleanup the user accepted. Requires: #2, #3, #4.

### Scope

- Delete `radiologist-core/.../registry/pull.py` and the upload/link half of `promote.py`; drop `pull_checkpoint` and `promote_to_registry` from `radiologist.core.registry.__init__` and `core.__all__`.
- Remove `pull_model` from `radiologist-inference/.../predictor.py`; rewire `Predictor.from_registry` to use `WandbRegistry().pull`/the registry instead.
- Update each affected package's `registry` extra to depend on `radiologist-registry`.
- Update package READMEs / CLAUDE.md references to the moved APIs.
- **Not in scope**: new behaviors, predictor refactor (separate epic), changing resolution semantics.

### Acceptance criteria

- [ ] `from radiologist.core import ...` no longer exposes `pull_checkpoint` or `promote_to_registry`; `from radiologist.inference import ...` no longer exposes `pull_model`.
- [ ] `Predictor.from_registry` still loads a predictor from a registry artifact path (behavior preserved through the new registry).
- [ ] All pre-existing tests pass without behavioral modification — only import paths/wiring change. If a test asserting old-symbol behavior must change, it is replaced by the equivalent behavioral test against `WandbRegistry`, not deleted silently.
- [ ] mypy clean; pytest green across all packages.
