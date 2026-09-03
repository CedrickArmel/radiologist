---
name: feedback-shared-base-class-seam-in-decomposition-epics
description: when a slice issue's public API needs a shared base-class constructor/helper still stubbed by a "parallel" sibling issue, implement the minimal real version yourself rather than leaving it NotImplementedError
metadata:
  type: feedback
---

In "decompose monolith into one-class-per-file hierarchy" epics (e.g. `BasePredictor` →
`Classifier`/`Explainer`, plus `MCDropoutPredictor(BasePredictor)`), a slice issue's brief
often says "not your job: base class core (issue #N), runs in parallel" — but the slice's
own acceptance bar ("no `NotImplementedError` reachable through `Subclass.from_path(...).method(...)`")
requires that same base-class constructor and its private helpers (`_read_metadata`,
`_preprocess_image`, etc.) to actually work.

**Why:** in a real two-agent parallel run the other issue's worktree would fill that gap
before merge. In a solo session there is no concurrent agent — if you leave the base-class
stub as `NotImplementedError`, your own tests cannot reach GREEN-real, which violates the
harder, more concrete instruction (GREEN-real bar) in favor of a softer scope note ("not your
job"). Port only the exact helpers your acceptance criteria touch (verbatim from the
monolith when one exists) and leave unrelated stubs (e.g. `from_registry`,
`_apply_prior_correction` when your slice doesn't use priors) untouched — don't implement
the sibling issue's full scope, just the minimum shared seam your own public API needs.

**How to apply:** when a stub in a shared base class blocks your slice's GREEN-real bar,
grep the monolith (e.g. `predictor.py`) for the equivalent logic and port it verbatim into
the base class, scoped to only what your contract requires. Flag in your final report that
you touched shared infra beyond your named files, so the reviewer/orchestrator knows to
reconcile it against the sibling issue's branch. See [[project_pipeline_architecture]] for
this repo's general ops/prefect split precedent of "implement the seam, not the whole
neighbor."
