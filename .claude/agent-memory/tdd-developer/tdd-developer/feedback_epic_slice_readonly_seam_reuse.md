---
name: epic-slice-readonly-seam-reuse
description: when a parallel epic slice's contract says a collaborator (e.g. observe_error) is "already implemented", verify by reading the file instead of assuming a gap to fill
metadata:
  type: feedback
---

An issue body may hedge ("verify this is implemented; if not, it's a gap left by
a prior issue — fix it as part of this slice's contract") about whether a shared
seam built by an earlier issue in the same epic is complete. Before writing any
extra code for that seam, read the actual file first — the earlier issue may
already have shipped the full, correct behavior (e.g. `Metrics.observe_error`
silently ignoring an `error_type` outside a closed set was already fully
implemented by the prior issue in this epic, contrary to the hedge language).

**Why:** Assuming a gap and "completing" already-correct code risks duplicating
guards or drifting from the original contract, and wastes a red/green cycle on
code that isn't actually the slice's scope.

**How to apply:** For any epic slice whose issue body says "verify X is
implemented; if not, fix it", read X's source first. Only touch it if it is
truly incomplete. Report explicitly in the summary which case applied.
