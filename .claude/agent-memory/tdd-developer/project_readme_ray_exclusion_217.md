---
name: readme-ray-exclusion-217
description: Issue #217 (Epic 2) — radiologist-etl README now documents that [all] excludes Ray on purpose, with opt-in and CI/dev-setup mechanism explained
metadata:
  type: project
---

Issue #217, part of Epic 2 (Milestone #19, [[shared/MEMORY.md]]), added an
"In `all`?" column to the execution-runner extras table in
`radiologist-etl/README.md` and a new `#### What all installs (and why Ray is
not in it)` subsection right after it, before the pre-existing `#### Beam`
subsection.

**Why:** `radiologist-etl[all]` deliberately excludes `ray` (execution family
still under development, #188) while including Beam (shipped, #189) — without
this doc, a user who installs `[all]`, picks `runner=ray_local`, and hits an
"install the ray extra" error can't tell it's expected, not a bug.

**How to apply:** the mechanism is two-layer and easy to conflate — `all`'s
own composition excludes `ray` in `pyproject.toml` (issue #214), while
`make dev-install`/CI exclude it via `--all-extras --no-extra ray` (issues
#215/#216), a *separate* exclusion because `--all-extras` bypasses `all`'s
composition and installs every named extra directly. `make docs-install`
intentionally has no such exclusion — `mkdocstrings` needs every optional
module importable. Sibling issues #214/#215/#216 land the actual flag changes
on parallel branches off the same epic branch; this issue is prose-only and
touches no build file.
