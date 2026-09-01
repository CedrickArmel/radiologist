---
name: prefect-broken-local-server-use-fn-bypass
description: In this repo's dev container, real @flow/@task execution needs a live Prefect API; the local ephemeral server is broken (Starlette version mismatch) and real Prefect Cloud creds are present in env — use Prefect's own .fn escape hatch to test real business logic without either.
metadata:
  type: project
---

In this project's dev/worktree containers, the installed Prefect (3.7.4 at
time of writing) is incompatible with the resolved Starlette release: any
plain call to a `@flow`- or `@task`-decorated function that needs to spin up
Prefect's local ephemeral API server fails with
`AttributeError: 'PrefectRouter' object has no attribute 'routes'` (a genuine
third-party version mismatch, not a bug in `radiologist.etl`).

Separately, this container's env also has **real Prefect Cloud credentials**
(`PREFECT_API_URL`/`PREFECT_API_KEY` pointing at `api.prefect.cloud`) —
calling `etl_flow(cfg)` (or any decorated flow) **without** popping these
vars will silently succeed by hitting the real hosted account. That's why
`radiologist-etl/radiologist_etl_tests/conftest.py` pops both vars for every
test — don't remove that, and always pop them (or use `env=` with them
stripped) in any subprocess-based test too. Never let a test hit real cloud.

This is why `radiologist-etl/radiologist_etl_tests/test_pipelines.py` stays
`--ignore`d in root `pyproject.toml` — it calls `etl_flow(cfg)` directly and
hits the broken local server once cloud creds are stripped.

**Working pattern** (used in `test_prefect_pipelines.py` and
`test_etl_command.py`): call the underlying plain function via Prefect's own
documented `.fn` attribute — `some_flow.fn(cfg)` / `some_task.fn(...)` —
which runs 100% real business logic with zero HTTP/orchestration calls. Also
stub the module's own artifact-creation calls
(`create_link_artifact`/`create_markdown_artifact`/`create_table_artifact`)
to no-ops via `monkeypatch.setattr` — these are true HTTP-boundary calls (not
owned business logic) and always try to reach the API even under `.fn`.
Pattern:

```python
for task_name in ("compute_stats_task", "apply_filters_task", ...):
    monkeypatch.setattr(prefect_pipelines, task_name, getattr(prefect_pipelines, task_name).fn)
monkeypatch.setattr(prefect_pipelines, "create_markdown_artifact", lambda **_: None)
result = prefect_pipelines.etl_flow.fn(cfg)
```

**Why**: `.fn` bypasses only Prefect's orchestration/tracking layer (a true
process boundary — HTTP calls to the Prefect API), never the actual
`radiologist.etl` logic, so it doesn't violate "never mock owned code."

**Gotchas when calling a `@hydra.main`-decorated function in-process
(not subprocess) more than once per pytest session**: clear
`hydra.core.global_hydra.GlobalHydra` before *and* after each call (it
raises "GlobalHydra is already initialized" on the second call otherwise).
Also always override `hydra.run.dir=`/`hydra.sweep.dir=` to a `tmp_path`
subdir — the default composed config (`radiologist-etl/.../conf/etl.yaml`)
points `hydra.run.dir` at the repo root by default, littering it with
`outputs/<date>/<time>/` dirs on every real invocation.

**Also watch the default config's `masks_root`**: it defaults to a real
`gs://radiologist-liora-gcs/...` bucket. Any test that runs the real
pipeline via Hydra composition (not passing an explicit `OmegaConf.create`
dict) must override `masks_root=null` or it will try (and fail, or worse,
succeed against a real bucket) a GCS call.

See also [[feedback_worktree_shell_chaining_blocked]] for the Bash sandbox
rules that made debugging this (multiple throwaway `python -c` scripts)
slower than it should have been — always write scratch scripts via the
`Write` tool to a path inside the worktree, never inline heredocs in Bash.
