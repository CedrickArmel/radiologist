## 🐛 Flow wiring: zero-valued knobs are ignored and the build report artifact is overwritten

**Requires:** #1, #2 · **Blocks:** #10
**Sole owner of the flow-wiring module for this epic**

### Context

The three ETL stage flows translate a Hydra config into stage-function
arguments. Three defects live in that translation layer, all in one file. They
are grouped because two of them land inside the same thirty-line flow body and
splitting them would serialise three developers through one function for no
benefit.

**Defect A — truthiness instead of null-checking.** Four call sites pull an
optional integer out of the config with the shape
`int(cfg.k) if OmegaConf.select(cfg, "k") else None`. `OmegaConf.select` returns
the *value*, so a configured `0` is falsy and silently becomes `None`, which the
stage then replaces with its own default. Configuring `batch_size=0` or
`workers=0` — an explicit, deliberate, invalid value — is silently swallowed
instead of being honoured or rejected. The sibling parameters in the very same
call already do the right thing with an explicit `is not None` (`iqr_columns`,
`iqr_factor`, `max_failure_rate` at `prefect_pipelines.py:322-338`), which is
what makes this an inconsistency rather than a design choice.

**Defect B — two artifacts, one key.** The build flow creates a table artifact
carrying the split report and a link artifact carrying the output directory, and
gives them the **same** key. Same key means successive versions of one artifact,
so the link — created second — wins and the split-report table is never
surfaced. The operator loses the only rendered view of the configured-versus-
observed split distribution.

**Defect C — the build failure tolerance is not wired.** #2 adds a
`max_failure_rate` key to `conf/build.yaml` and a matching parameter to
`build_shards`, but this file owns every flow-to-stage argument, so until this
issue lands the key is inert. The build flow's completion artifact also does not
report the failure count that #2 makes available.

### Steps to reproduce

**A:** Run the extract flow with `batch_size=0`. Observed: the run proceeds
using the default batch size of 64, with no error and no warning.

**B:** Run the build flow with prefect installed and inspect the run's
artifacts. Observed: one artifact under the build run's key, containing the
output link. The split-report table is absent.

**C:** Set `max_failure_rate: 0.5` in the build stage's config and run a build
in which a minority of images are unreadable. Observed (after #2, before this
issue): the run still fails, because the configured tolerance never reaches the
stage.

### Expected vs actual

**Expected:** an explicitly configured value is either used or rejected with a
clear error, never silently replaced. A completed build run surfaces both its
split-report table and its output link as distinct artifacts. The build stage's
configured failure tolerance takes effect.

**Actual:** zero is swallowed; the table artifact is shadowed; the tolerance is
dropped.

### Root cause

`radiologist-etl/src/radiologist/etl/prefect_pipelines.py` — the four
truthiness sites, at exactly these lines:

```python
# :283  extract_flow
batch_size = int(cfg.batch_size) if OmegaConf.select(cfg, "batch_size") else None
# :332  extract_flow, inside the extract_stage(...) call
workers=int(cfg.workers) if OmegaConf.select(cfg, "workers") else None,
# :435  build_flow, inside the build_shards_stage(...) call
workers=int(cfg.workers) if OmegaConf.select(cfg, "workers") else None,
# :475  run_extract
batch_size = int(cfg.batch_size) if OmegaConf.select(cfg, "batch_size") else None
```

and the colliding artifact keys in `build_flow` at `:449-462`:

```python
create_table_artifact(
    table=rows,
    key=f"build-{result.run_id}",
    description=f"Shard split report for run {result.run_id}",
)

create_link_artifact(
    link=result.output_dir,
    key=f"build-{result.run_id}",
    description=(
        f"Build output for run {result.run_id}: "
        f"{result.shard_count} shard(s), {result.record_count} record(s)."
    ),
)
```

Note that `build_flow` calls `resolve_execution(OmegaConf.select(cfg, "runner"))`
at `:413-417` with **no** `batch_size` argument, and `conf/build.yaml` has no
top-level `batch_size` key — so the build flow's truthiness site is the
`workers` one only. There is no build-flow batch-size defect.

### Behaviour to implement

1. **A private optional-int accessor.** Introduce a module-private helper in
   this file:

   ```python
   def _opt_int(cfg: DictConfig, key: str) -> int | None:
       # contract: returns int(value) when the key resolves to a non-null value,
       # None when the key is absent or explicitly null. A configured 0 returns 0.
   ```

   Use it at all four sites above. This is a **private** helper, not a new
   public name: it exists because the identical four-fold expression is what
   broke, and collapsing it to one site is what stops it breaking again. Do not
   export it, do not move it to another module, do not generalise it to other
   types — the `float` siblings in the same calls are already correct and stay
   inline.

2. **Zero is honoured, therefore rejected loudly.** With the accessor in place,
   `batch_size=0` reaches `chunked`, which already raises
   `ValueError(f"size must be >= 1, got {size!r}")` (`execution.py:102-103`), and
   `workers=0` reaches `ProcessPoolExecutor(max_workers=0, ...)`, which already
   raises. That is the correct outcome: an explicit invalid value produces a
   clear error at the point it is used, rather than a silent substitution. Do
   not add a new guard to "improve" it — the existing errors already name the
   value, and adding a second guard would duplicate a check.

3. **Distinct artifact keys.** Give the build flow's table artifact its own key,
   `build-report-{run_id}`. Leave the link artifact's key as `build-{run_id}` so
   the identifier operators already use for a build run keeps pointing at the
   output link.

4. **Wire the build failure tolerance.** `build_flow` reads `max_failure_rate`
   from its config with the same `is not None` discipline `extract_flow` already
   uses for the same key at `:334-338`, defaulting to `0.0`, and passes it to
   `build_shards_stage`. The build flow's link-artifact description reports the
   failure count alongside the existing shard and record counts.

### Acceptance criteria

- [ ] Running the extract flow with a batch size explicitly configured as `0`
      raises an error naming the invalid size, rather than completing with the
      default batch size.
- [ ] Running the extract flow with a worker count explicitly configured as `0`
      raises, rather than completing with the default worker count.
- [ ] Running the build flow with a worker count explicitly configured as `0`
      raises, rather than completing with the default worker count.
- [ ] Running a stage with a batch size or worker count that is absent, or
      explicitly null, completes using that stage's documented default —
      unchanged from today.
- [ ] Running the extract flow with a batch size explicitly configured as a
      positive integer dispatches work in batches of exactly that size.
- [ ] A completed build run produces two distinct artifacts: one carrying the
      split-report rows and one carrying a link to the output directory. Neither
      shadows the other, and the two are created under different keys.
- [ ] Running the build flow with a failure tolerance configured above the
      observed failure rate completes, and the run's reported failure count
      equals the number of records that could not be written into a shard.
- [ ] Running the build flow with a failure tolerance configured below the
      observed failure rate fails with a build-failure error.
- [ ] Running the build flow with no failure tolerance configured behaves as if
      it were `0.0`.
- [ ] The description of a completed build run's output-link artifact names the
      failure count.
- [ ] mypy clean; pytest green

### Out of scope

- The `float`-valued optional parameters (`iqr_factor`, `max_failure_rate`),
  which already use an explicit null check and are not changed except where the
  new build-flow read is added.
- Changing the link artifact's key.
- Adding a `batch_size` key to the build stage's config.
- Any change outside this module.

### Technical notes

- `radiologist-etl/src/radiologist/etl/prefect_pipelines.py` — this issue is the
  sole owner of this file for the epic. Nothing else in the epic edits it.
- The `max_failure_rate` parameter on `build_shards` and the `BuildFailureError`
  type were added by #1 and given behaviour by #2. The `failed` field on the
  build result was added by #1. This issue only passes the configured value
  through and reports the resulting count.
- **No existing test pins either artifact key literal.** A repository-wide search
  for `build-{` in `radiologist-etl/radiologist_etl_tests/` returns nothing;
  the flow tests assert on `link`, `description` and `table` payloads only. The
  key change is therefore purely additive — no existing test needs to change for
  this issue.
- The flow tests in `radiologist-etl/radiologist_etl_tests/test_prefect_pipelines.py`
  call the flows through their `.fn` escape hatch (for example
  `extract_flow.fn(cfg, execution=plan)`) because the real Prefect engine does
  not run in this sandbox. Follow that convention.
- An autouse fixture in that module stubs the three artifact-creating functions
  by monkeypatching them **on the `prefect_pipelines` module**
  (`monkeypatch.setattr(prefect_pipelines, "create_link_artifact", lambda **_: None)`
  and the same for `create_markdown_artifact` / `create_table_artifact`);
  individual tests override one of them with a capturing callable to inspect its
  keyword arguments. Use that mechanism to observe the two artifact keys — these
  are true process boundaries (the Prefect SDK), so stubbing them is allowed.
  Everything importable as a local module must be reached through the public API,
  never mocked.
- The same test module installs spy doubles over `extract_batch_task` and
  `write_shard_task` to verify that mapped work is submitted in bounded waves;
  the batch-size criterion above is naturally expressed against that mechanism.
- Note that `prefect_pipelines.py` imports `_PREFECT_AVAILABLE` **by value**
  (`:35-45`), so tests toggle it with
  `monkeypatch.setattr(prefect_pipelines, "_PREFECT_AVAILABLE", False)`, not on
  the `optional` module. Never use `pytest.mark.skipif`.
- Docs: this fix invalidates no documented behaviour — the `build`
  `max_failure_rate` config-table row is added by #2. No doc changes here.

### Design notes

The private `_opt_int` helper is the only piece of shared structure this issue
introduces, and it clears the "same defect in two or more places" bar four times
over. It is deliberately kept module-private rather than promoted to
`execution.py` next to `_cfg_get`: that accessor serves runner *nodes* and has
its own null-handling contract (fixed in #5), while this one serves *stage*
config with an `int` cast and an `OmegaConf.select` lookup. Merging them would
couple two files that this epic has otherwise kept under separate ownership, for
a saving of about six lines.
