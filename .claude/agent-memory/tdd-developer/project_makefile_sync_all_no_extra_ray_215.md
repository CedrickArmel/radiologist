---
name: makefile-sync-all-no-extra-ray-215
description: Issue #215 (Epic 2) — sync-all's --no-extra ray flag alone is not sufficient until #214 also lands; verified via uv sync --dry-run
metadata:
  type: project
---

`uv sync --active --all-groups --all-packages --all-extras --no-extra ray` (root
Makefile `sync-all` target) excludes the *named* `ray` extra, but as of this
issue (#215, before #214 merges) `radiologist-etl`'s `all` extra still directly
lists `prefect-ray` as a hard dependency (not composed from the `ray` extra) —
see `radiologist-etl/pyproject.toml`'s `[project.optional-dependencies]`. So a
dry-run (`uv sync ... --all-extras --no-extra ray --dry-run`) still shows
`+ prefect-ray` / `+ ray` in the plan on this branch, because `--all-extras`
also enumerates the `all` extra itself, whose direct dependency list bypasses
`--no-extra`.

**Why:** `--no-extra <name>` (uv >= 0.7.13) excludes a *named* optional-deps
group; it does not prune a package that a *different* named extra (`all`)
happens to also list directly. Issue #215's own text anticipates this: it says
the fix "is not blocked on #214" and "remains necessary after it lands" —
meaning full exclusion of ray requires BOTH #214 (make `all` compose from the
narrower extras instead of duplicating `prefect-ray`) AND #215 (the Makefile
flag) landed together. Confirmed empirically: `uv pip list --active | grep ray`
showed `prefect-ray==0.5.0` / `ray==2.58.0` installed after `make dev-install`
with the #215 fix alone, on a branch without #214.

**How to apply:** don't be alarmed if an issue's own literal AC ("leaves the
Ray backend distribution uninstalled") doesn't fully hold when verified in
isolation on a branch that only contains one of two complementary fixes in the
same epic — check the issue's "out of scope" section for language like "not
blocked on X, remains necessary after it lands" as the signal that full
verification requires the sibling branch's merge too. Implement exactly the
frozen scope; don't reach into the sibling issue's file to make your own AC
pass in isolation.

See also [[project_etl_extras_scoping_epic_219]] (if it exists) and
[[feedback_oom_kills_pytest_under_parallel_agents]].
