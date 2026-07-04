---
name: epic-seam-convention-ownership-move
description: when a new shared-seam issue's interface contract says a naming/rewrite convention now lives in the new module, check whether an existing sibling method already applies that same convention internally — it must be simplified to stop double-applying it
metadata:
  type: feedback
---

In radiologist#142 (predictor-verb registry `verbs.py`), the interface contract required
`load_predictor` to rewrite `run_id` via `apply_mcd_convention` (`f"{run_id}-mcd"`) *before*
building the registry selector, for the `uncertainty` verb. But the pre-existing
`MCDropoutPredictor.from_selector` (delivered by a prior sibling issue, #141) already applied
that exact same `{run_id}-mcd` suffixing internally (plus an extra, no-longer-wanted pull of the
deterministic model) — a design left over from an older CLI-only bugfix (#139). Blindly wiring
`load_predictor` on top of the untouched `from_selector` double-applied the suffix
(`run1` → `run1-mcd` → `run1-mcd-mcd`), a bug only surfaced by writing a real registry-backed
test for the `uncertainty` verb (mocking only the wandb boundary) and reading the resulting
mock call args.

**Why:** epic specs describe the target end-state of the *new* module's contract, not every
existing collaborator that also happens to encode the same behavior today. The issue text's own
phrase "the deterministic model is no longer pulled" was the signal that a sibling method's
current behavior was slated to change as *part of* this slice's shared seam, not a fact I could
take as already true.

**How to apply:** when an issue's design notes state a convention "is exactly what `<new
function>` encodes" or "is no longer done automatically", grep for the old inline version of that
same logic in the classes/functions the new function will call into. If found, it is a shared
seam this slice must simplify (per
[[feedback_shared_base_class_seam_in_decomposition_epics]]) — remove the duplicate rewrite from
the old location and update its existing tests to reflect the new single-application contract,
rather than leaving both layers applying the convention.
