---
name: etl-all-drops-ray-214
description: issue #214 (Epic 2) removed prefect-ray from radiologist-etl's `all` extra, leaving the standalone `ray` extra untouched
metadata:
  type: project
---

Issue #214 (milestone #19, Epic 2 "Scope radiologist-etl's default extras
away from deferred execution backends") removed `"prefect-ray>=0.4.0"` from
`radiologist-etl`'s `all` extra in `radiologist-etl/pyproject.toml`, with a
comment pointing at #188 (Ray execution family still under development). The
five single extras (`gcs`, `prefect`, `dask`, `ray`, `beam`) were left
untouched — `radiologist-etl[ray]` still resolves `prefect-ray`.

TDD: extended the frozen contract test module from #213
(`radiologist_etl_tests/test_packaging_extras.py`, which parses
`pyproject.toml` via `tomllib`/`tomli`, no import of `radiologist.etl`) with
one new test, `test_all_extra_excludes_the_deferred_ray_backend`, asserting
`_extra_distribution_names(data, "all") & _extra_distribution_names(data,
"ray")` is empty. Watched it fail (`found {'prefect-ray'}`) before editing the
TOML.

Sibling issues #215 (Makefile) and #216 (CI workflows) fix the separate
`uv sync --all-extras` bypass in parallel — out of scope here. `uv.lock`
needed regenerating (`uv lock`, no `--active` flag exists for that
subcommand) and picked up an unrelated `tomli` marker addition from #213's
own pyproject change that hadn't been locked yet — harmless, not part of this
diff's intent.

See also [[feedback_oom_kills_pytest_under_parallel_agents]] and
[[feedback_spawn_pool_needs_pythonpath_single_file]] for how the full
`radiologist-etl` suite was verified green despite heavy concurrent-agent
memory contention on this box.
