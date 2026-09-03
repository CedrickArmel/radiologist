---
name: project-pipeline-architecture
description: ETL ops.py/prefect.py split pattern, compute_run_id contract, Prefect 3 API facts
metadata:
  type: project
---

## ops.py / prefect.py split

Portable pipeline logic lives in `ops.py` — pure Python functions with zero Prefect imports.
Prefect orchestration shells live in `prefect.py` — `@task` / `@flow` decorators that delegate entirely to `ops.py` functions; no logic inside the decorated bodies beyond calling the core and creating artifacts.

`ops.py` is importable without `prefect` installed. `prefect.py` guards all Prefect imports with `try/except ImportError` stub no-ops.

Use `OmegaConf.select(cfg, "key")` (not `cfg.get("key")`) for optional `DictConfig` keys. Cast `OmegaConf.to_container()` result to `dict()` explicitly to satisfy mypy.

## compute_run_id contract

`run_label` is always included in the SHA-256 fingerprint, never returned verbatim.
Same label + same data → same ID (idempotent). Different label → different ID (forces new artifacts).

**Why:** prevents silent artifact overwrite when `run_label` is reused across different dataset versions.

**How to apply:** Any future change to `compute_run_id` must preserve this hashing behaviour. Tests verify both divergence (`label != no_label`) and stability (`same label twice → same ID`).

## Prefect 3 API facts

- `create_link_artifact` is **not** a coroutine — call it synchronously inside task bodies.
- `cache_policy=INPUTS` (from `prefect.cache_policies`) enables task-level idempotency.
- Calling `etl_flow(cfg)` directly without a runner spins up a temporary Prefect server inline — valid for testing.
