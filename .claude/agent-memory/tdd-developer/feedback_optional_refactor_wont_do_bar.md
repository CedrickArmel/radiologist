---
name: optional-refactor-wont-do-bar
description: When an optional refactor issue states its own "not an improvement" threshold (e.g. max parameter count), apply it literally and report won't-do rather than forcing the extraction
metadata:
  type: feedback
---

When an issue is marked **optional** and its technical notes pre-commit to a
threshold for what counts as a non-improvement ("a helper with four parameters
and two conditional branches is not an improvement"), draft both the inline and
the extracted version, count against that threshold, and if the extraction
exceeds it, close as **won't-do and say so**. That is the sanctioned outcome,
not a failure to deliver.

**Why:** the architect writes these thresholds precisely because they cannot see
the merged code at spec time and are delegating a judgement call. Forcing an
extraction that the author already declared not-an-improvement produces churn
and a worse call site than the duplication it replaces. The orchestrator's
prompt may still read as directive ("your job is to extract the shared logic")
— the *issue body* is authoritative on this point and usually also asks whether
the predicted duplication actually materialised.

**How to apply:** count the parameters the helper genuinely needs after reading
the *merged* code, not the code the issue was written against. Sibling issues
that landed in between routinely add axes of variation the issue did not
predict (an extra descriptive phrase, an interposed log call), pushing a
predicted 3-axis difference to 4+. Report the drafted helper verbatim in your
summary so a human can overrule cheaply, and name the extra axes you found.

Related: [[feedback_review_fix_reverts_epic_seam_scope_creep]],
[[feedback_epic_seam_convention_ownership_move]].
