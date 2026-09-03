---
name: worktree-branch-already-checked-out
description: git refuses a second worktree for a branch already checked out elsewhere — work directly in the checkout that has it instead
metadata:
  type: feedback
---

When asked to amend an existing feature branch and the task instructions say
"create a worktree for this", first check whether the primary checkout (or
any existing worktree) already has that branch checked out
(`git worktree list`, `git branch --show-current`). `git worktree add
<path> <branch>` fails with "already used by worktree at ..." if so — git
disallows checking the same branch out twice.

**Why:** this happened on issue #139 (amending
`feat/129-configurable-preprocessing-normalization`): the main checkout at
`/workspaces/radiologist` was already on that branch from prior work.
Attempting `git worktree add .claude/worktrees/139-selector-fix
feat/129-configurable-preprocessing-normalization` failed immediately.

**How to apply:** when this happens, skip the worktree/dedicated-venv setup
entirely and just work in the checkout that already has the branch, using
its already-synced venv (activate via `pyenv activate <existing-venv>`, no
need for `make dev-install` if deps are already synced — verify with a quick
`uv sync --active` no-op check or just try running tests first). Don't force
a worktree that git structurally can't create. Still follow all other rules
(no direct commits to main, Commitizen messages, etc.) — only the
worktree/venv isolation step is skipped, because it's redundant when the
branch is already isolated to that one checkout.

Also noteworthy: `make dev-install` / `uv sync --active` inside a
repo can non-destructively regenerate `uv.lock` with a different `revision`
field (e.g. uv version skew) even with no dependency changes — check
`git diff uv.lock` before committing and `git checkout -- uv.lock` if the
diff is just noise unrelated to the task.

See also: [[feedback_pyenv_activate_needs_shell_init]].
