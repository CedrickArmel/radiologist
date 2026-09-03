---
name: etl-correctness-epic-decisions
description: Binding user decisions for the radiologist-etl ten-defect correctness epic (pragmatic-balance blueprint), settled 2026-09-02 — do not reopen them in follow-up design work
metadata:
  type: project
---

The `radiologist-etl` ten-defect correctness epic was designed on 2026-09-02.
The user compared minimal-impact / clean-architecture / pragmatic-balance and
chose **pragmatic-balance**: 1 skeleton + 8 slices + 1 optional refactor.

Settled decisions — restate, never re-litigate:

- **Rejected abstractions:** no `layout` / `settings` / `failures` modules, no
  `ExecutionBackend` / `Runner` protocol. Multi-backend execution rides
  Prefect's `TaskRunner` + Hydra `_target_` (see
  [[feature_prefect_native_runner_selection]]).
- **Beam scratch parts are reclaimed on success AND on failure** (`try/finally`).
  Competing designs proposed retaining parts for post-mortem; explicitly
  rejected because a failed large Dataflow run is exactly the case that leaks
  ~15.6k objects and never gets swept. Cleanup failure logs a WARNING naming the
  prefix and never masks the original error.
- **Both prefect gates relaxed:** `resolve_execution`'s `local` family and the
  CLI's `require("etl")`. The `etl` group becomes module-importability-only,
  like `inference`. `radiologist-cli` is in scope.
- **Full extract parity for build failures:** `max_failure_rate` config key +
  parameter, a `BuildFailureError`, `failed`/`failure_rate` on `BuildResult`, and
  shard-write-failed records marked `excluded=True` so "no non-excluded record
  has `shard=None`" holds.
- **The assign/build run-id break happens exactly once**, in one issue, with an
  operational-impact notice.
- **Broader ETL doc rot is out of scope** and deserves its own follow-up epic:
  `docs/reference/config-etl.md` and root `README.md` both still document the
  retired monolithic `etl_flow` / `conf/etl.yaml` pipeline.

**Why:** the user reviewed three competing blueprints and picked one explicitly;
re-proposing the rejected structure wastes a review cycle.

**How to apply:** when asked to extend, re-slice, or implement any of these ten
defects, treat the above as fixed constraints. Design work is limited to detail
inside the chosen slicing.

Final blueprint lives at `/tmp/etl-fixes-final/` (ephemeral — regenerate if gone).
Related: [[feedback_deferred_issues_stay_additive]],
[[feedback_no_protocol_null_object_ceremony]].
