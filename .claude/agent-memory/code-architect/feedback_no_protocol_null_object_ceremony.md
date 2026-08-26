---
name: no-protocol-null-object-ceremony
description: User rejects Protocol + Null-Object pairs and extra module splits for internal single-implementation features; one concrete class with an internal enabled flag
metadata:
  type: feedback
---

For an **internal** feature that will only ever have one real implementation, do not propose
a `Protocol` with a real + null implementation pair, and do not split it across an extra
module just because the layers are conceptually distinct. Ship **one concrete class** in
**one new module**, with an internal `_enabled` flag for the degraded case.

**Why:** stated when choosing between two architectures for Prometheus metrics in
`radiologist-inference` (2026-08-26). The user picked the single-file design over a
`MetricsRecorder` Protocol + `PrometheusMetrics`/`NullMetrics` pair + a separate pure-ASGI
`middleware.py`, citing the repo's own rule against abstractions beyond what the task
requires and against designing for hypothetical futures. The app will never have a second
metrics backend and will never be served by anything but FastAPI, so the seam bought
nothing and cost a file, a Protocol and a class.

**How to apply:** when a design instinct says "extract an interface so it's swappable" or
"split the transport layer out so it's framework-agnostic", first ask whether a second
implementation or a second framework is actually coming. If not, inline it — the FastAPI
middleware goes directly in `_build_app`, the recorder is one class. Reserve Protocols for
places where two real implementations already exist. Related:
[[optional-feature-gating]], [[per-application-resource-lifetime]].
