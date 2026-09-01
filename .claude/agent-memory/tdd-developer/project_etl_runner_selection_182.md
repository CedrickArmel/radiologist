---
name: etl-runner-selection-182
description: issue #182 (three-stage ETL epic, milestone #16) runner-selection deliverable — ExecutionPlan/resolve_execution contract and conf/runner/*.yaml shape that #186 (Prefect orchestration) consumes
metadata:
  type: project
---

Implemented issue #182 on branch `feat/182-runner-selection` (based on
`feat/16-etl-three-stage-framework`, not `main`), commit `b9bae70`.

**What #186 (Prefect orchestration layer) can rely on:**

- `radiologist.etl.execution.resolve_execution(runner_cfg, batch_size=None) ->
  ExecutionPlan` is real, GREEN-real code — no `NotImplementedError` reachable
  through it. `ExecutionPlan` is a frozen 3+1-field dataclass
  (`family`, `task_runner`, `beam`, `batch_size`). Dispatch on `family` (or
  simpler: `plan.task_runner is not None` vs `plan.beam is not None`) is the
  whole wiring decision — no Beam-specific type is named anywhere in
  `execution.py`, so #189 (deferred Beam issue) never needs to reopen it.
- Local/Dask/Ray all resolve through `hydra.utils.instantiate(runner_cfg.task_runner)`;
  Beam resolves through `hydra.utils.instantiate(runner_cfg.beam)` into a real
  `BeamExecutor` (constructor only — `run_batches` is still `NotImplementedError`,
  owned by #189).
- Availability gating: `_backend_available(family)` checks
  `radiologist.etl.optional._PREFECT_AVAILABLE` /
  `_PREFECT_DASK_AVAILABLE` / `_PREFECT_RAY_AVAILABLE` / `_BEAM_AVAILABLE`
  *before* calling `instantiate`, so a missing extra raises `RuntimeError`
  naming `radiologist-etl[<family>]`, never a raw `ImportError`/Hydra
  instantiation traceback.
- `conf/runner/local.yaml` now carries `batch_size: 64` (was missing from the
  skeleton). `dask_local/dask_address/dask_cluster.yaml` are fully wired
  per the issue body's literal snippets. `ray_local/ray_cluster.yaml` and
  `beam_direct/beam_dataflow.yaml` compose cleanly under Hydra (verified by
  a parametrized test) but their backends are inert — Ray issue #188 and
  Beam issue #189 only need to touch `beam_executor.py`'s `run_batches` /
  add real Ray wiring notes, not any file this issue owns.
- `radiologist-etl` pyproject gained `dask`, `ray`, `beam` extras (folded
  into `all`), each a single third-party package
  (`prefect-dask`/`prefect-ray`/`apache-beam`).

**Test strategy for backends not installed in this environment**
(prefect_dask/prefect_ray/apache_beam are absent from the shared venv and
disk was too full to install them — 1.2G free): mocked the *third-party SDK*
itself, not owned code — `monkeypatch.setitem(sys.modules, "prefect_dask",
<fake module with a stub DaskTaskRunner class>)` combined with
`monkeypatch.setattr(radiologist.etl.optional, "_PREFECT_DASK_AVAILABLE",
True)`. This drives `resolve_execution`'s real logic and `hydra.utils.instantiate`
for real, only stubbing the boundary package — consistent with "mock only
true process boundaries / 3rd-party SDKs," not the local `execution.py` code.
Local family didn't need this — `prefect` itself (with `ProcessPoolTaskRunner`)
is a real, already-installed dependency.

**Left untouched in `execution.py`** (still `NotImplementedError` stubs,
correctly out of this issue's scope): `default_workers`, `chunked`,
`local_mapper` — these back the `BatchMapper`/`ShardMapper` local-pool
default and are #183 (extract)/#185 (build) territory, not runner
*selection*. Do not implement them under a future issue believing #182
already covers "execution.py" wholesale — it only owns `ExecutionPlan` +
`resolve_execution`.

See also [[etl-three-stage-skeleton-180]] for the sibling skeleton
decisions this issue built on top of, and the disk-starved shared-venv
testing pattern that also applied here — verified again:
`/home/vscode/.pyenv/versions/radiologist/bin/python -m pytest
radiologist-etl/radiologist_etl_tests --confcutdir=.` works cleanly from a
worktree without corrupting the shared venv's `.pth` files, since no `uv
sync --active` was run.
