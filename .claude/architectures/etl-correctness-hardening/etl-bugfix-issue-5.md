## 🐛 The shipped default runner refuses to run on a plain install

**Requires:** #1 · **Blocks:** —
**Spans two packages on purpose (see Design notes)**

### Context

The ETL package documents prefect as *optional*. `radiologist-etl/README.md`'s
dependencies section states that when prefect is not installed "`@flow` and
`@task` are identity decorators and the pipeline runs as plain Python", and the
same file's execution-runner table lists the extra required for the `local`
family as **none**.

It does not work that way. Three gates make a plain, no-extras install unusable:

**Gate 1 — the runner-availability predicate.** The `local` execution family is
reported as available only when prefect is importable. Both
`radiologist-etl/src/radiologist/etl/conf/extract.yaml` and
`.../conf/build.yaml` declare `defaults: [_self_, {runner: local}]`, so running
either stage on the **shipped defaults** raises a runtime error.

**Gate 2 — the error message names an extra that does not exist.** When a family
is unavailable, the message interpolates the family name into the prose and a
lookup result into the install command. For `local` the lookup misses and falls
back to `"prefect"`, so the message reads "the **local** extra is required to
use the local runner family. Install with: pip install
`radiologist-etl[prefect]`" — inconsistent with itself. And there is no `local`
extra: `radiologist-etl`'s declared extras are exactly `gcs`, `prefect`, `dask`,
`ray`, `beam`, `all`.

**Gate 3 — the CLI's own extra check.** `require("etl")` raises unless the ETL
package's prefect-availability sentinel is true, independently of the library.
So even with gates 1 and 2 fixed, `radiologist etl ...` would still refuse to
start. The `inference` branch of the same helper already gates on module
importability only; `etl` becomes consistent with it. **This is a settled user
decision: both gates are relaxed — `resolve_execution`'s `local` family and the
CLI's `require("etl")`.**

A fourth hazard sits behind gate 1: `conf/runner/local.yaml` declares
`task_runner._target_: prefect.task_runners.ProcessPoolTaskRunner`. If the
availability gate is simply removed, resolution would try to instantiate that
target and fail with an import error instead. The `local` family must resolve to
*no* task runner when prefect is absent — which is exactly what the flows
already handle, since `with_task_runner` only attaches a runner when
`_PREFECT_AVAILABLE` and `plan.task_runner is not None`.

**A separate, unrelated defect in the same function's call path:** the config
accessor returns `None` for a key that is present but explicitly null, because
it forwards to a `get(key, default)` that only substitutes the default for
*absent* keys. Configuring `runner.batch_size=null` therefore produces an
execution plan with a null batch size, which reaches the chunking helper via
`extract(batch_size=plan.batch_size)` and dies with
`TypeError: '<' not supported between instances of 'NoneType' and 'int'`. It is
fixed here because it lives in the same accessor, in the same function's call
path, and splitting it out would put two developers in one twenty-line region.

### Steps to reproduce

1. Install the workspace with no optional extras.
2. Run `radiologist etl extract file_list=... destination=...`.
3. Observed: the command fails before doing any work, with a message offering to
   install a `radiologist-etl[local]` extra that does not exist.
4. Separately, with prefect installed, resolve a runner configuration whose
   `batch_size` is explicitly `null` and run a stage under the resulting plan.
5. Observed: `TypeError: '<' not supported between instances of 'NoneType' and
   'int'`.

### Root cause

`radiologist-etl/src/radiologist/etl/execution.py:188-207` and `:250-255`:

```python
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if getter is not None:
        return getter(key, default)      # returns None for an explicit null
    return getattr(cfg, key, default)


def _backend_available(family: str) -> bool:
    if family == "local":
        return bool(optional._PREFECT_AVAILABLE)     # <- gate 1
    if family == "dask":
        return bool(optional._PREFECT_DASK_AVAILABLE)
    ...

_BACKEND_EXTRAS = {"dask": "dask", "ray": "ray", "beam": "beam"}
...
if not _backend_available(family):
    extra = _BACKEND_EXTRAS.get(family, "prefect")    # <- "local" misses -> "prefect"
    raise RuntimeError(
        f"the {family} extra is required to use the {family} runner family. "
        f"Install with: pip install 'radiologist-etl[{extra}]'"   # <- gate 2
    )
```

`radiologist-cli/src/radiologist/cli/optional.py:78-82`:

```python
if extra == "etl":
    from radiologist.etl import optional as etl_optional

    if not etl_optional._PREFECT_AVAILABLE:      # <- gate 3
        raise RuntimeError(hint)
```

### Behaviour to implement

1. **`local` is always available.** The availability predicate returns true for
   the `local` family unconditionally. `dask`, `ray` and `beam` keep their
   existing sentinel checks.
2. **Never instantiate a task runner that cannot be imported.** When the family
   is `local` and prefect is not importable, resolution must return a plan with
   `family="local"` and `task_runner=None`, without evaluating the configured
   `task_runner` target. With prefect importable, the shipped `local` runner
   config must still be instantiated exactly as it is today. Do **not** edit
   `conf/runner/local.yaml` — it is correct for the prefect-installed case.
3. **Emit a followable remedy.** The unavailable-backend message must name an
   extra that exists in the package's metadata. Since `local` can no longer be
   unavailable, the remedy set now covers exactly the families that can be:
   `dask`, `ray`, `beam` — all three of which are already keys in
   `_BACKEND_EXTRAS`. Keep the message shape ("the X extra is required to use
   the X runner family. Install with: pip install `radiologist-etl[X]`") and
   make the prose and the command name the same extra.
4. **A present-but-null key falls back to its default.** The config accessor
   returns the supplied default when the looked-up value is `None`, not just
   when the key is absent. This applies uniformly to every key it reads
   (`family`, `batch_size`, `task_runner`, `beam`).
5. **The CLI gates the `etl` group on importability only.** Remove the
   prefect-sentinel branch from `require()`, so `etl` behaves like `inference`
   already does: present and importable is enough. The `registry` branch's W&B
   sentinel check is untouched. Update the `require()` docstring, which
   currently documents `etl` as gating on
   `radiologist.etl.optional._PREFECT_AVAILABLE`.

### Acceptance criteria

- [ ] With prefect not importable, resolving the shipped default runner
      configuration yields an execution plan for the local family and does not
      raise.
- [ ] With prefect not importable, resolving the shipped default runner
      configuration yields a plan carrying no task runner, and the configured
      task-runner target is never instantiated.
- [ ] With prefect importable, resolving the shipped default runner
      configuration yields a plan carrying an instantiated task runner —
      unchanged from today.
- [ ] Resolving a runner configuration for a backend family whose package is not
      importable raises an error whose install command names an extra that the
      package actually declares, and whose prose names the same extra as the
      command.
- [ ] Resolving a runner configuration whose batch size is explicitly null
      yields a plan whose batch size is the documented default of `64`, and a
      stage run under that plan completes rather than raising a type error.
- [ ] Resolving a runner configuration whose family key is explicitly null still
      yields a plan for the local family — a regression pin, unchanged from
      today.
- [ ] With prefect not importable, requiring the `etl` command group returns the
      ETL module rather than raising, and dispatching `radiologist etl` with no
      subcommand prints its usage line.
- [ ] With the ETL package not importable at all, requiring the `etl` command
      group raises an error naming `pip install 'radiologist-cli[etl]'`.
- [ ] With the W&B SDK unavailable, requiring the `registry` command group still
      raises an error naming `pip install 'radiologist-cli[registry]'` —
      unchanged from today.
- [ ] mypy clean; pytest green

### Existing test this issue supersedes

**One existing test asserts the gate this issue removes and must be rewritten as
part of this issue.** In
`radiologist-cli/radiologist_cli_tests/test_optional_require.py`:

```python
def test_raises_cli_hint_not_business_hint_when_etl_feature_sentinel_missing(
    self, monkeypatch
) -> None:
    from radiologist.cli.optional import require
    from radiologist.etl import optional as etl_optional

    monkeypatch.setattr(etl_optional, "_PREFECT_AVAILABLE", False)

    with pytest.raises(RuntimeError) as excinfo:
        require("etl")
    ...
```

After this fix `require("etl")` returns the module instead of raising. Replace
this test with one asserting the corrected contract at the same public surface:
with the prefect sentinel false, `require("etl")` returns the ETL module. The
sibling test that patches `radiologist.cli.optional._etl` to `None` and expects
the `radiologist-cli[etl]` hint must be added if it does not already exist — the
existing absent-package test covers `inference`, not `etl`. The registry test in
the same file is untouched. These are the **only** existing tests this issue may
touch.

### Out of scope

- Making the `dask`, `ray` or `beam` families work without their packages.
- Any change to `conf/runner/local.yaml` or the other runner configs.
- The flows' own prefect-absence warning, which already behaves correctly.
- The `registry` group's W&B sentinel check.

### Technical notes

- `radiologist-etl/src/radiologist/etl/execution.py` — this issue is the sole
  owner of this file for the epic.
- `radiologist-cli/src/radiologist/cli/optional.py` — the module already imports
  `radiologist.etl` inside a `try/except ImportError` at module scope and stores
  `None` on failure (`:38-41`); that stored value, checked at `:74-76`, *is* the
  importability check the `etl` branch collapses onto. Removing the branch is a
  deletion, not a rewrite.
- **Sentinel patching — the important nuance.** `execution.py` does
  `from radiologist.etl import optional` (`:46`) and reads
  `optional._PREFECT_AVAILABLE` at *access* time (`:200`), so
  `monkeypatch.setattr(radiologist.etl.optional, "_PREFECT_AVAILABLE", False)`
  does affect it. By contrast `prefect_pipelines.py` imports the flag *by value*
  (`from radiologist.etl.optional import _PREFECT_AVAILABLE`, `:35-45`), so
  patching the `optional` module does **not** affect that file — its own tests
  patch `prefect_pipelines._PREFECT_AVAILABLE` directly. For this issue you want
  the `optional`-module form. Never use `pytest.mark.skipif`, and never mock
  owned code.
- To assert that the task-runner target is not instantiated when prefect is
  absent, observe the resolved plan's `task_runner` rather than patching
  `hydra.utils.instantiate` — a plan with no task runner is the observable
  consequence, and a plan built from a config whose target is unimportable can
  only be produced by not instantiating it.
- Note that `family = _cfg_get(runner_cfg, "family", "local") or "local"`
  (`execution.py:239`) already collapses an explicit null family to `"local"`
  today. The corresponding acceptance criterion is a regression pin on existing
  behaviour, not a new fix — do not remove the `or "local"` while changing
  `_cfg_get`.
- Docs, `radiologist-etl/README.md`: the dependencies paragraph already states
  that prefect is optional and that without it the pipeline "runs as plain
  Python", and the execution-runner table already lists the extra for
  `local (default)` as "none". Both statements are correct *documentation of
  intent* that the code violates — this fix makes them true. Amend only if a
  clause is needed to make explicit that the default `local` runner requires no
  extra. Touch nothing else in that file — #2, #3 and #8 own other sections of
  it and are landing concurrently.
- Docs, `radiologist-cli/README.md`: the extras table maps the `etl` group to
  `radiologist-etl[all]`. Since the group now works on a plain install, amend
  only that row to say the `etl` extra is needed for the optional execution
  backends, not to start the group.

### Design notes

Three gates and one config-accessor bug are grouped into a single issue because
they are a single **user-visible defect**: "the pipeline does not run on a plain
install." Fixing the library gate without the CLI gate leaves `radiologist etl`
broken; fixing the CLI gate without the library gate leaves `run_extract`
broken. A reviewer asked to confirm "does a plain install work now?" needs to
see all of it in one diff. The null-key accessor bug is folded in because it
lives inside the same function's call path — the alternative is two developers
editing a twenty-line region concurrently, which costs more in conflict
resolution than it buys in parallelism.

The rejected alternative was an `ExecutionBackend` protocol with per-family
availability and instantiation policies. Rejected: multi-backend execution here
deliberately rides Prefect's own `TaskRunner` abstraction plus Hydra `_target_`
instantiation, and this defect is a one-line predicate error — a protocol would
add an abstraction to fix a typo.
