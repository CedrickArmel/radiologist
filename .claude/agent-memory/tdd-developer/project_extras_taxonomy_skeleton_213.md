---
name: project-extras-taxonomy-skeleton-213
description: Epic 2 skeleton issue #213 froze the extras-taxonomy contract and shipped test_packaging_extras.py; #214-217 implement the actual ray-exclusion diffs
metadata:
  type: project
---

Issue #213 (Epic 2 milestone #19, "Scope radiologist-etl's default extras
away from deferred execution backends") is a skeleton/contract issue: it adds
`radiologist-etl/radiologist_etl_tests/test_packaging_extras.py`, which parses
`radiologist-etl/pyproject.toml` with `tomllib`/`tomli` and pins three
invariants that already hold today — each of gcs/prefect/dask/ray/beam is a
single-distribution extra, `[project] dependencies` names no orchestrator or
backend package, and `all` is a superset of gcs/prefect/dask/beam. It
deliberately does **not** assert `ray ∉ all` — that assertion belongs to
#214, red-first.

**Why this matters for reviewing #214-#217**: this issue only touched the
root `pyproject.toml` (added `"tomli>=2.0.1; python_version < '3.11'"` to the
`test` dependency-group, since it was previously only in the `release`
group — see [[feedback_cz_uv_provider_cwd_relative_lockfile]] for other tomli
context) and the new test file. `radiologist-etl/pyproject.toml`, the
`Makefile`, `ci.yml`/`publish.yml`, and the README were read but left
byte-identical — those are #214-#217's job per the frozen contract in the
issue body.

**How to apply**: when implementing #214 (drop `ray` from `all`), the new
test file's `_PRODUCTION_READY_EXTRAS` superset check will still pass (ray
was never in that tuple) — #214 needs its own new red assertion
(`ray not in all_names`) rather than relying on this file catching the
regression.
