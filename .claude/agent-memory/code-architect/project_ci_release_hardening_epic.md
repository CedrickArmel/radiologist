---
name: ci-release-hardening-epic
description: Confirmed scope decisions for the CI/build/release hardening epic, incl. the deliberate deferral of path-filtered CI and its activation trigger
metadata:
  type: project
---

The CI/build/release hardening epic (spec drafted 2026-09-06) was designed against
scope decisions the user pre-confirmed — do not relitigate these when revisiting:

- `UV_FROZEN: "true"` wherever `uv sync` runs in CI; a `.github/actions/setup-uv`
  composite action is the single place it is set, with a `frozen:` input for opt-out.
- TestPyPI dry-run + a `uv pip compile --no-sources` pin-resolution guard both gate the
  real PyPI push in `publish.yml`.
- Minimum-version resolution testing is **scheduled/nightly only**, never on PRs.
- Cross-package pin cascade is **advisory only** — it annotates the release PR body and
  must never fail a release.
- **Path-filtered / modular CI is deliberately NOT built.** Activation trigger: workspace
  exceeds ~10 members, OR `ci.yml`'s test job wall-clock exceeds ~15 min. Today: 6
  members, comfortably inside budget. A 7th (`radiologist-app`) is planned.

**Why:** the repo's real failure mode is declared metadata drifting from reality across a
six-directory monorepo, not CI slowness. Path filtering on a single shared lockfile trades
correctness risk for time the repo does not yet need.

**How to apply:** design new tooling to derive facts from the workspace manifest rather
than restate them, and keep `ci.yml` jobs single-concern so they can later be lifted into
`workflow_call` reusable workflows without a rewrite. See
[[feedback-derive-dont-restate-workspace-roster]].
