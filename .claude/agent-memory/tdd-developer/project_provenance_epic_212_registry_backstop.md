---
name: provenance-epic-212-registry-backstop
description: Epic 1 issue #212 (registry resolve_selector path-required backstop + docs) — shape of the guard and why the CLI regression tests still passed without new prod code
metadata:
  type: project
---

Issue #212 (milestone #18, "Explicit registry provenance") added a guard to
`radiologist.registry.selector.resolve_selector`: after the existing
`run_id and tags` mutual-exclusion check, if `selector.is_registry_backed()`
and `selector.path.strip()` is empty, raise `ValueError(_PATH_REQUIRED_MSG)`.
`is_registry_backed()` was deliberately left untouched — it only answers
"did the caller ask for a registry lookup", not "is the lookup well-formed".

**Why this mattered**: `radiologist registry resolve`/`promote` already took
`path` as a mandatory positional `typer.Argument(...)`, so a CLI regression
test invoking them with no path argument passes even without the new guard —
that's just Typer's own required-arg check. The guard's real bite is a
blank/whitespace `--path " "` combined with a selector flag (Typer sees the
arg as present, non-empty), or any future caller that skips the CLI (e.g.
`registry pull`, or a library user calling `resolve_selector` directly).
The test that actually exercises the new code is `pull` invoked with
`path="   "` plus `--run-id` — assert `mock_wandb.Api.assert_not_called()`.

Docs updated in the same issue: `docs/reference/cli-inference.md`'s
"Registry selector vs. local path" section had stale mutual-exclusivity
prose left over from before #209/#210 (which now *require* `--path` together
with a selector, since `--path` supplies entity/project) — rewrote it plus
all four commands' `--run-id` table rows and "Emitted keys" lines to mention
`model_qualified_name`/`model_version` (added by the concurrent #211 slice,
not yet merged when #212 was written — docs describe the target state).
`radiologist-registry/README.md`'s `resolve_selector` description also
needed a second validation clause.

See also [[project_provenance_epic_github_issues]].
