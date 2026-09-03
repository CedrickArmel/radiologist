---
name: fastapi-prefect-cap-192
description: fastapi is capped workspace-wide via [tool.uv] constraint-dependencies to keep Prefect's embedded local API server working
metadata:
  type: project
---

Root `pyproject.toml` now has a `[tool.uv]` table with
`constraint-dependencies = ["fastapi<0.116.0"]` (added for GitHub issue
#192, fixed in commit e2f3be3 on `fix/192-prefect-fastapi-router-incompatibility`).

**Why:** Prefect 3.7.x ships its own `PrefectRouter` (FastAPI `APIRouter`
subclass) to back its local/ephemeral API server (the one used whenever
`PREFECT_API_URL` is empty/unset — e.g. under [[feedback_prefect_broken_local_server_use_fn_bypass]]-style
local test isolation). FastAPI's internal routing/route-matching-cache
implementation changed between 0.115.0 and 0.137.1 in a way `PrefectRouter`
was never updated for, causing `AttributeError: 'PrefectRouter' object has
no attribute 'routes'`. Prefect's own declared range (`fastapi<1.0.0,>=0.111.0`)
is wide enough for `uv` to legitimately resolve past what `PrefectRouter`
actually works with. Capping fastapi (not starlette — Prefect genuinely
needs `starlette>=1.0.1`, downgrading it alone breaks the local server
differently) and letting the resolver pick the paired Starlette naturally
is the fix. Resolved after the cap: `fastapi==0.115.14`, `starlette==0.46.2`,
`prefect==3.7.2` (was `0.137.1`/`1.3.1`/`3.7.4`).

**How to apply:** if a future `uv lock` regression reintroduces this
symptom, check whether the fastapi cap in root `pyproject.toml` is still
present/tight enough before re-diagnosing from scratch. A regression
canary lives at
`radiologist-etl/radiologist_etl_tests/test_prefect_engine_canary.py` —
it runs a real (non-`.fn`) `@flow` against the forced-local-server path,
so this exact class of bug fails there specifically instead of scattered
across the suite.
