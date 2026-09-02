---
name: etl-cli-cutover-187
description: issue #187 (three-stage ETL epic, milestone #16) CLI cutover — subcommand dispatch pattern in radiologist-cli/groups/etl.py and the "local runner default needs a live Prefect engine" test gotcha #190 (final refactor) should know about
metadata:
  type: project
---

Implemented issue #187 on branch `feat/187-cli-cutover` (based on
`feat/16-etl-three-stage-framework`), commit `853b7f9`.

**Subcommand dispatch shape in `radiologist-cli/src/radiologist/cli/groups/etl.py`**:
`SUBCOMMANDS = ("extract", "assign-split", "build")`; `run(argv)` calls
`extract_output_flag` on the *full* argv first (so `--output` works on
either side of the subcommand token — load-bearing per the issue body),
then pops the leading token. No token / unknown token -> usage on stderr,
nonzero. `--help`/`-h` with no valid subcommand -> usage on stdout, zero.
A valid subcommand dispatches to `extract_main`/`assign_split_main`/
`build_main` (each its own `@hydra.main` entry point), with `sys.argv`
rewritten, `RADIOLOGIST_OUTPUT` saved/set/restored in try/finally, and
`SystemExit` translated to a plain int (mirrors the retired single-command
`run()` exactly). One shared `_ensure_input_exists(uri, label,
storage_options)` helper (fsspec-based) backs all three stages' precondition
checks — `cfg.file_list` / `cfg.manifests_dir` (folder) / `cfg.split_manifest`.

**`radiologist.etl` package exports**: `etl_flow`, `EtlResult`,
`StatsProcessor`, `compute_run_id`, and the five `*_task` orchestration
wrappers (`apply_filters_task`, `assign_splits_task`, `build_shards_task`,
`compute_stats_task`, `write_jsonl_task`) were removed from
`radiologist/etl/__init__.py`'s imports and `__all__` — but their *code*
still lives in `ops.py`/`prefect_pipelines.py`/`processors.py`, untouched.
Per the issue body, only the CLI's/package's *exposure* is cut over here;
actually deleting the dead code is explicitly deferred to #190 ("retire the
private stage-operation indirection layer"). `conf/etl.yaml` (with its
`resume_from_*` keys) *was* deleted in this issue (the config file itself,
unlike the .py code, was explicitly in scope). `radiologist-etl/radiologist_etl_tests/test_pipelines.py`
(already `--ignore`d in root `pyproject.toml`, now removed along with that
ignore line) and the two `etl_flow`/`EtlResult`-specific tests inside
`test_prefect_pipelines.py` were deleted; that file's other ~30 tests
(covering `extract_flow`/`assign_split_flow`/`build_flow`/`run_extract`/etc
from #186) were untouched and still pass.

**Gotcha: `runner=local`'s default `ProcessPoolTaskRunner` breaks real
`.fn()`-bypass tests in this environment.** `conf/runner/local.yaml` sets a
*real* `task_runner`, not `None` — so even the default (no explicit
`runner=` override) causes `_extract_batch_mapper`/`_shard_mapper` to build
a `.map()`-based closure once `resolve_execution` runs, which needs a live
Prefect flow-run context this environment's broken ephemeral server can't
provide (see [[feedback_prefect_broken_local_server_use_fn_bypass]]). Any
CLI-level test that wants to exercise the *real* end-to-end stage via the
`_FlowSpy(.fn)` bypass pattern must pass the Hydra deletion override
`"~runner"` (not `"runner=null"` — that leaves the key present with value
null in a way that didn't test cleanly; `~runner` removes the config-group
node entirely so `OmegaConf.select(cfg, "runner")` is `None` and
`resolve_execution(None)` returns `task_runner=None`). Tests that instead
want to verify *which* runner family gets selected (the "runner override
selects the family" AC) should keep `runner=local`/`runner=ray_local` etc.
but swap the flow object for a `_FlowSpy` wrapping a **no-op** lambda (not
the real `.fn`) — exactly mirroring
`test_run_extract_resolves_the_plan_and_attaches_its_task_runner` in
`radiologist-etl/radiologist_etl_tests/test_prefect_pipelines.py` — since a
real attached task runner can never actually run stage logic here.

**Gotcha: `prefect_dask` IS installed** in this repo's shared venv (unlike
what `[[etl-runner-selection-182]]`'s memory assumed at the time of #182) —
`runner=dask_local` resolves successfully. Only `prefect_ray` and
`apache_beam` are genuinely absent, so the "runner backend not installed"
AC test must use `runner=ray_local` (asserting `"radiologist-etl[ray]"` in
stderr), not `dask_local`. Re-check installed extras with
`python -c "import prefect_dask"` rather than trusting old memory before
picking which family to use for this kind of test.

**Gotcha: subprocess-based CLI tests need explicit `PYTHONPATH` in a
worktree.** `subprocess.run([sys.executable, "-c", ...], cwd=CLI_ROOT)`
does *not* inherit pytest's `conftest.py` `sys.path.insert` shim — the
spawned interpreter falls back to the shared venv's editable-install `.pth`
files, which point at the **main checkout**, not the worktree. Any CLI
subprocess helper (`_run_cli_subprocess` here) must set
`env["PYTHONPATH"]` to the worktree's `radiologist-*/src` dirs explicitly,
or it will silently exercise stale main-checkout code and produce
confusing failures (e.g. `AttributeError: module has no attribute
'extract_main'` when the module actually does define it — just not the one
that got imported). Direct in-process calls (`etl_group.run(...)` inside
the pytest process itself, run via `--confcutdir=.`) are unaffected and
resolve correctly.

See also [[etl-three-stage-skeleton-180]], [[etl-runner-selection-182]],
[[feedback_prefect_broken_local_server_use_fn_bypass]].
