---
name: feedback-review-fix-reverts-epic-seam-scope-creep
description: when a code-review finding says a helper module's names leaked into a package __init__/__all__, the fix is a straight revert (remove import + names), not a re-negotiation of the seam
metadata:
  type: feedback
---

On PR #144 (predictor-verb-registry epic), review flagged that
`radiologist-inference`'s `__init__.py` had re-exported `verbs.py`'s
`PredictorVerb`/`get_verb`/`apply_mcd_convention`/`load_predictor` into
`__all__`, directly contradicting the epic issue's own explicit instruction
("Do not edit `__init__.py` — verbs stays internal"). This is the same class
of thing as [[feedback_shared_base_class_seam_in_decomposition_epics]] and
[[feedback_skeleton_issue_repoints_old_tests]]: an internal CLI-support seam
had scope-crept into the public API surface during implementation, and the
review caught it after merge.

**Why:** `cli.py` already imported `verbs` directly
(`from radiologist.inference import verbs`) — nothing needed the re-export.
Once one file re-exports an internal seam, the package's exact-equality
`__all__` test (`test_public_api.py::test_all_public_names_present`) has to
grow with it, silently normalizing the leak for future reviewers/agents who
just diff against the "current" expected set instead of the epic's original
intent.

**How to apply:** when a finding says "X leaked into `__all__`", revert both
sides atomically in one commit: remove the import block + the names from
`__all__` in `__init__.py`, AND shrink the test's expected `set` literal back
to what it was before the leaking commit (check `git log -p` on the test file
if unsure what "before" means) — don't just make the test pass by keeping the
leaked names in both places. A second finding on the same PR (redundant
`MCDropoutPredictor.from_selector` override that had become byte-for-byte
identical to the inherited `BasePredictor.from_selector` after
`_resolve_and_pull` absorbed the registry-fallback logic) is the same root
cause in miniature: a decomposition epic's shared-seam code drifting out of
sync between sibling/slice issues. Also: this PR's `TestServeCommand` mocked
`create_app` (owned code) — reviewed fix was to let the real `create_app`
run (it's side-effect-free once `_uvicorn.run` is mocked as the actual
process boundary) and assert through FastAPI's `TestClient`, mirroring the
pattern already established in `test_app.py`.
