---
name: dedupe-fix-ripples-across-test-doubles
description: eliminating a redundant SDK call inside a shared resolve/pull seam breaks every test double built to distinguish the two calls, in every package that duplicates that helper
metadata:
  type: feedback
---

When a fix removes a redundant call to a shared boundary (e.g. `_WandbResolver.pull()`
skipping its own `api.artifact()` when `resolve()` already fetched the same
qualified name via an instance-scoped cache), any test double built around the
old two-call shape breaks — not just in the package that owns the seam, but in
every consuming package that hand-rolled an equivalent double.

Concretely: `radiologist-registry`'s `_WandbResolver` had a `resolve()` →
`pull()` path where both called `api.artifact()` independently. Fixing #221
to cache-and-reuse the resolved artifact broke a `_make_registry_wandb_mock()`
helper duplicated verbatim across `radiologist-inference/.../test_verbs.py`,
`radiologist-cli/.../test_inference_commands.py`, and
`radiologist-cli/.../test_serve_command.py`. All three used
`api.artifact.side_effect` returning a different mock for the kwargs-style
resolve() call vs. the positional pull() call, then configured `.download()`
only on the "pulled" mock — which the fix now never touches.

**Why:** grepping only the target package's tests before declaring green
misses consuming packages (inference, cli) that copy-pasted the same
process-boundary-mocking pattern instead of importing a shared fixture.

**How to apply:** after any change to a shared resolve/pull/fetch seam,
`grep -r` the distinguishing pattern (e.g. `side_effect.*kwargs` or the
helper's name) across *all* workspace packages' test dirs, not just the
package whose source changed. Update every duplicate consistently: point the
"downloaded" mock at the object that's actually used post-fix, and add a
`call_count == 1` assertion so the dedup itself is asserted, not just
tolerated.

See also: [[project_provenance_epic_212_registry_backstop]],
[[feedback_issue_test_scope_incomplete]].
