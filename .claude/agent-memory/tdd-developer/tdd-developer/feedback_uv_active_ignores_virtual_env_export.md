---
name: uv-active-ignores-virtual-env-export
description: uv run/sync --active inside a worktree can resolve to the shared pyenv venv despite VIRTUAL_ENV/PATH being exported, corrupting its .pth files
metadata:
  type: feedback
---

`uv run --active` / `uv sync --active`, invoked from inside a worktree whose
`.python-version` names an unsupported/mismatched pyenv version string, can
silently fall back to resolving the **shared** `radiologist` pyenv venv
(`~/.pyenv/versions/3.10.16/envs/radiologist`) even when `VIRTUAL_ENV` and
`PATH` are exported pointing at the worktree's dedicated venv in the same
Bash call. `--active` trusts the currently-activated env, and pyenv's shim
resolution for the `uv` binary itself can override that — the failure mode
is silent (no error), and only surfaces later as edits to `radiologist_*.pth`
files under the shared venv's `site-packages/`, each rewritten to point at
the worktree path.

**Why:** caught this rewriting all 7 `radiologist_*.pth` files (+
`direct_url.json` metadata) in the shared venv to point at
`.claude/worktrees/<name>/...` after a single `uv run --active pytest
--collect-only`. Recovered by hand-editing each `.pth`/`direct_url.json`
back to `/workspaces/radiologist/<pkg>/src` — a plain string swap since these
are simple one-line path files, not compiled state. `uv.lock` mtime also
moved but content hash was unchanged (false alarm, safe).

**How to apply:** never pass `--active` to `uv` from inside a worktree.
Instead run `uv sync --python <absolute path to the dedicated pyenv
virtualenv's python3> --all-groups --all-packages --all-extras` from the
worktree root. Note this does **not** actually install into that pyenv
virtualenv — uv creates its own self-contained `.venv/` in the worktree
directory instead (the `--python` flag only picks the base interpreter).
That `.venv/bin/python3 -m pytest ...` is what to use for all test/mypy
invocations for the rest of the session; the pyenv virtualenv named after
the worktree ends up unused but still gets deleted per the normal
worktree-cleanup step. Before trusting any shared venv is undamaged after
an accidental `--active` run, `grep -rl <worktree-name>
~/.pyenv/versions/3.10.16/envs/radiologist/lib/python3.10/site-packages/`
to check for leaked absolute paths.

See also [[shared/MEMORY.md]].
