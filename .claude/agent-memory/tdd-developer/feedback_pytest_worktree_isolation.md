---
name: feedback-pytest-worktree-isolation
description: pytest invocation gotchas in this monorepo — stale editable-install .pth entries and cross-package plugin name collisions
metadata:
  type: feedback
---

In this monorepo, editable installs (`*.pth` files in a pyenv virtualenv's
site-packages) are plain sys.path-appending files pointing at whichever
worktree last ran `make dev-install` / `uv sync` in that venv. If a venv is
ever reused across worktrees (e.g. before running `make dev-install` in a
freshly created worktree, or if the per-worktree venv setup in CLAUDE.md's
[Worktrees](#worktrees) section was skipped), a package's `.pth` file can
point at a *different* (possibly since-deleted) worktree, not the one
currently being edited.

The root `conftest.py` sys.path shim is supposed to make each worktree
self-consistent regardless of stale `.pth` entries, but it only takes effect
when pytest resolves conftest collection from the actual rootdir. Passing an
explicit package-scoped path (e.g. `pytest radiologist-core/tests`, exactly
what `make test-core` does under the hood) can silently skip loading the
top-level `conftest.py`, so imports fall through to site-packages and quietly
resolve to a different worktree's source — tests can pass or fail against
code you never touched, with no visible error.

**Why:** discovered while implementing issue #109 — `test_resume_onnx_skeleton_contract.py`
tests failed with "DID NOT RAISE NotImplementedError" even though the stub in
this worktree still raised, because Python was importing `radiologist.core.resume`
from a sibling agent's worktree via a stale `.pth` file, and the sibling had already
implemented that function.

**How to apply:** in a worktree, always invoke pytest with `--confcutdir=.`
pinned to the repo root alongside the explicit package path, e.g.:
`python -m pytest radiologist-core/tests --confcutdir=. -q`. This forces
pytest to load the root `conftest.py` (which inserts this worktree's own
`src/` dirs at the front of `sys.path`), so imports resolve to the current
worktree regardless of what the venv's `.pth` files point to. Do NOT
run bare `python -m pytest` with no path args across all 5 testpaths at once —
each package's `tests/conftest.py` registers under the same plugin name
`tests.conftest` (no `__init__.py` in test dirs), causing a
`ValueError: Plugin already registered under a different name` when more than
one package's tests collect in the same session. Run one package's tests dir
at a time with `--confcutdir=.`.

If your own worktree's dedicated venv (per CLAUDE.md's per-worktree pyenv
setup) is correctly created and `make dev-install` has been run in it, this
whole class of problem shouldn't arise — this is a fallback diagnosis for
when tests resolve against unexpected code, not a routine workaround.
