---
name: per-application-resource-lifetime
description: Stateful process-global registries must be built per create_app() call, not at module import, because the test suite constructs one app per fixture
metadata:
  type: feedback
---

Any stateful third-party registry/singleton (Prometheus `CollectorRegistry`, metric
collectors, caches, connection pools) must be constructed **inside the application factory**
— one instance per `create_app()` call — never at module import onto a shared/global
object.

**Why:** confirmed as the chosen design for `radiologist-inference` metrics (2026-08-26).
The inference test suite builds a fresh app per pytest fixture, per test. A module-level
registry makes every count shared across apps: tests cross-contaminate, counts become
test-order dependent, per-app assertions are impossible, and moving construction into the
factory later raises `Duplicated timeseries in CollectorRegistry` on the second call. Two
independent design passes both flagged this as the load-bearing constraint.

**How to apply:** hold the per-app instance in the factory's existing closure state rather
than inventing new plumbing — in `_build_app` that means adding a key to the existing
`state_holder` dict next to `"predictor"`. Accept the trade-off explicitly (a per-app
registry rules out `prometheus_client.multiprocess`, i.e. uvicorn `--workers > 1`) and
document it. Related: [[no-protocol-null-object-ceremony]].
