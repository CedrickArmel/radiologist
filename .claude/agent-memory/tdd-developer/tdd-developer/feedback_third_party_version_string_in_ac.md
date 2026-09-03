---
name: third-party-version-string-in-ac
description: An issue's acceptance criterion can quote a literal third-party output string (e.g. an exact Content-Type version) that drifts from the version actually pinned in the lockfile
metadata:
  type: feedback
---

An issue/spec can give a literal example of a third-party library's output
(e.g. "Content-Type: text/plain; version=0.0.4; charset=utf-8" for
`prometheus_client`) that was accurate for an older release but not for the
version actually resolved by the lockfile. Example: `prometheus_client`
0.25.0 emits `version=1.0.0`, not `0.0.4`, in `CONTENT_TYPE_LATEST`.

**Why:** the spec author is describing intent ("the response announces the
Prometheus text format"), not literally pinning a wire-format version. If the
implementation already uses the library's own constant
(`client.CONTENT_TYPE_LATEST`) per the interface contract, that constant *is*
correct — the AC's embedded example is stale, not the code.

**How to apply:** when a test built from an issue's literal example string
fails only because of a version-number mismatch against a real dependency,
don't chase the exact string. Verify what the installed library actually
produces (`python -c "import <lib>; print(<lib>.CONTENT_TYPE_LATEST)"` or
equivalent), and assert against the library's own constant/contract instead
of a hardcoded literal from the issue text. Keep the *behavioral* assertion
(status 200, correct media type family, TYPE lines present) — drop the
brittle exact-version substring.
