---
name: worktree-venv-pth-corruption-risk
description: uv sync/dev-install inside a worktree can silently rewrite the shared pyenv venv's .pth files unless VIRTUAL_ENV is exported in every single Bash call, not just PATH
metadata:
  type: feedback
---

Setting only `PATH` to prefix a dedicated worktree venv's `bin/` is not enough to make
`uv sync --active` / `uv run --active` target that venv. `--active` reads the `VIRTUAL_ENV`
env var specifically; without it, uv falls back to resolving the project's `.python-version`
(which may alias to the shared `radiologist` pyenv env) and happily reinstalls/rewrites
editable `.pth` files there — exactly the cross-worktree corruption [[per_application_resource_lifetime]]-adjacent
CLAUDE.md gotcha warns about, silently, with no error.

**Why:** because agent shell tool calls do not persist environment state between invocations
(each Bash call is a fresh shell), an `export VIRTUAL_ENV=...` from a prior call does not carry
over — every single command that runs `uv sync --active` or `uv run --active` needs BOTH
`export VIRTUAL_ENV=<dedicated-venv-path>` and `export PATH="<dedicated-venv>/bin:$PATH"` in
the same command string. Forgetting `VIRTUAL_ENV` even once while `PATH` is set is enough to
have uv "successfully" sync into the wrong (shared) venv, rewriting its `.pth` files to point
into the worktree — which breaks the main checkout the moment the worktree is deleted.

**How to apply:** in any worktree session, prefix EVERY `uv sync`/`uv run --active`/pytest/mypy
invocation with both exports in the same Bash call:
`export VIRTUAL_ENV=/home/vscode/.pyenv/versions/<dedicated-venv> && export PATH="$VIRTUAL_ENV/bin:$PATH" && <command>`.
Also avoid `eval "$(pyenv init -)"` / `pyenv activate` inside this sandboxed tool — those get
blocked as "too complex to verify" by the worktree-isolation guard; the PATH+VIRTUAL_ENV
override is the workaround. After any suspected wrong-venv sync, check
`<shared-venv>/lib/python3.10/site-packages/*.pth` for paths containing `/worktrees/` — if
found, the shared venv's `.pth` files must be rewritten back to the main checkout's absolute
paths before finishing, or the main checkout's imports will break as soon as the worktree is
removed.
