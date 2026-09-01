---
name: unified-cli-centralization
description: Milestone #15 decision — all CLI code centralizes in a new radiologist-cli package; the four business packages become pure libraries with no console scripts
metadata:
  type: project
---

Milestone **#15** (`CedrickArmel/radiologist`, issues #170–#177) builds a single
`radiologist` console script. Decision confirmed by the user on **2026-08-29**, revising
an earlier design that had kept per-package CLIs behind a thin dispatcher:

- **All** command bodies live in a new `radiologist-cli` workspace member. `radiologist-etl`,
  `radiologist-core`, `radiologist-registry`, `radiologist-inference` become pure libraries —
  no `[project.scripts]`, no Typer app, no `@hydra.main` entry point.
- `radiologist-core[all]` is a **hard** dep of `radiologist-cli`; etl/registry/inference are
  **extras**. Root `radiologist[etl]` must point at `radiologist-cli[etl]`, because the user's
  stated intent is "installing an extra activates the *command*, not just the library".
- `radiologist-inference`'s existing `radiologist` script is **deleted, not renamed**.
- Cross-package Hydra composition uses `config_path="pkg://<module>"`. The user **empirically
  validated** this (help tree, `key=value` overrides, `--multirun`) before asking for the
  spec — treat it as settled and do not re-explore it.
- Issue **#177** (consolidate the duplicated `_exit_on_error`) is closed as obsolete:
  centralization means the duplication never comes into existence.

**Why:** four divergent output/exit conventions made the SDK unscriptable, and the user wants
one grammar decided in one place rather than a consolidation refactor later.

**How to apply:** when asked about milestone #15 or the CLI, assume centralization is the
baseline. Check whether the code has landed before asserting current state — as of
2026-08-29 nothing was implemented. See also [[optional-feature-gating]] (its
"no new extras" rule does *not* apply to workspace-member extras) and [[pytest-layout]]
(a sixth package means updating the root conftest shim, testpaths, and pyright includes).
