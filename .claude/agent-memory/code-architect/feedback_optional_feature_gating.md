---
name: optional-feature-gating
description: User rejects opt-in flags/params for optional features; gate on the existing extra's import sentinel and make the feature always-on when installed
metadata:
  type: feedback
---

When adding an optional capability (observability, integrations), do **not** propose a CLI
flag, an env var, or an `enable_x=` factory parameter, and do **not** create a new extra
for it. Add the dependency to the *existing* extra that already gates that surface (plus
the `all` aggregate), add an import sentinel to `optional.py`, and make the feature
always-on when the sentinel resolves — degrading silently (route simply not registered)
when it does not.

**Why:** stated directly while scoping Prometheus metrics for `radiologist-inference`
(2026-08-26) — the user explicitly rejected both an opt-in flag on `create_app()` and a
separate `metrics`/`monitoring` extra, wanting `prometheus-client` in the same `serve`
extra as fastapi/uvicorn. Rationale is that a knob nobody turns on is dead code, and the
extra already expresses the user's intent to serve.

**How to apply:** when designing any feature behind an optional dependency in this repo,
default to "always-on if importable". Propose configuration only if the user asks for it.
Keep call sites free of availability checks — but do it with an internal `enabled` flag on
one concrete class, **not** a Protocol + Null Object pair (see
[[no-protocol-null-object-ceremony]]). Only the route/middleware *registration* is
conditional, so the endpoint 404s cleanly when the extra is absent.

**Scope boundary (2026-08-29):** this rule is about *runtime capabilities inside a
package*, not about *which sibling workspace packages a distribution pulls in*. When the
optional thing is a whole workspace member, the user does want new extras — see
[[unified-cli-centralization]], where he asked for `etl`/`registry`/`inference` extras on
`radiologist-cli`. Don't quote this rule to argue against those.
