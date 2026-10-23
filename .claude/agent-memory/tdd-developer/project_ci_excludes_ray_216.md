---
name: project-ci-excludes-ray-216
description: issue #216 (Epic 2) added --no-extra ray to ci.yml/publish.yml test-job uv sync, tested via a text-parsing workflow test in scripts_tests
metadata:
  type: project
---

Issue #216 (milestone #19, Epic 2 "scope radiologist-etl's default extras
away from deferred execution backends") added `--no-extra ray` to the
`uv sync --active --all-groups --all-packages --all-extras` install step of
the `test` job in both `.github/workflows/ci.yml` and
`.github/workflows/publish.yml`, plus a two-line comment citing issue #188.
`.github/workflows/docs.yml` was deliberately left byte-for-byte untouched
(mkdocstrings needs every Hydra config target importable, Ray included).

**Why:** `--all-extras` ignores extra composition and force-installs
`prefect-ray` for the still-under-development Ray execution family even
though no test exercises it — the twin of issue #215 (`Makefile`) and #214
(`radiologist-etl/pyproject.toml`), same epic, different mechanism.

**How to apply:** the executable contract lives at
`scripts_tests/test_ci_workflows_exclude_ray.py` — a root-level, non-package
test directory (see `scripts_tests/test_release_bump.py` for precedent) that
is already in `[tool.pytest.ini_options] testpaths` in the root
`pyproject.toml`, so a new file there is picked up with zero config changes.
It parses the workflow YAML **as text** (line-scanning by 2-space-indent job
headers), not with `pyyaml` — `pyyaml` is only a transitive dependency here
(pulled in by prefect/commitizen/dask), not guaranteed installed on every
extras combination, and the skeleton test for issue #213
(`radiologist-etl/radiologist_etl_tests/test_packaging_extras.py`) sets the
same precedent of parsing config as text/`tomllib` rather than importing an
extra-only library. See also
[[project_extras_taxonomy_skeleton_213]].
