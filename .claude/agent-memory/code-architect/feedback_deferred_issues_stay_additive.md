---
name: deferred-issues-stay-additive
description: A deferred/phase-N issue must never reopen files an earlier issue delivered — put the generic dispatch branch in the earlier shared issue and verify it with a stand-in
metadata:
  type: feedback
---

When an epic defers a backend/variant to a later phase, the shared code it plugs into (flow
branch, dispatch switch, plan field) belongs to the **earlier** issue that owns that file — not
to the deferred issue. Write the branch against the *shape* of the shared record, naming no
type from the deferred family, and verify it with a stand-in object exposing the required
methods so the test does not depend on the deferred implementation existing.

**Why:** the user cross-checked the ETL three-stage spec and found issue 7 (orchestration)
declaring Beam "out of scope" while issue 10 (Beam) admitted it needed "one branch in each
flow" — which would have forced issue 10 to edit `prefect_pipelines.py`, contradicting the
Build Sequence table's framing that deferred phases are purely additive. An internal
contradiction between one issue's Out-of-scope list and another's Design Notes is exactly the
kind of drift that surfaces during implementation, not review.

**How to apply:** when writing an epic with deferred phases, read each deferred issue's Design
Notes against the earlier issues' Out-of-scope lists. If the deferred one says it costs "a
field plus a branch" somewhere, that branch must already be an acceptance criterion of whoever
owns the file. Also check the reverse: a family that genuinely needs zero shared-code changes
(Ray, a real Prefect TaskRunner) should say so explicitly and be left alone.

Related: [[prefect-native-runner-selection]], [[etl-three-stage-redesign]].
