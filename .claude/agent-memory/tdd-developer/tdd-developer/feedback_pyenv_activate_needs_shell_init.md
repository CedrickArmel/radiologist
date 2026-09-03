---
name: pyenv-activate-needs-shell-init
description: Bash tool calls need `eval "$(pyenv init -)"` and `eval "$(pyenv virtualenv-init -)"` before `pyenv activate` works — otherwise it silently no-ops and installs land in whatever venv the .python-version shim resolves to.
metadata:
  type: feedback
---

`pyenv activate <name>` in a fresh non-interactive Bash tool call fails with
"pyenv-virtualenv to be loaded into your shell" unless you first run
`eval "$(pyenv init -)"; eval "$(pyenv virtualenv-init -)"` in the same
command. If you skip this and the command "succeeds" anyway (no activation,
but no hard error either), `make dev-install`/`uv sync --active` will run
against whatever venv the ambient `.python-version` shim resolves to — in a
worktree, that shim commonly resolves back to the **shared** project venv,
so the install silently rewrites the shared venv's `.pth` files to the
worktree's absolute path. This is exactly the venv-corruption hazard
CLAUDE.md's worktree section warns about, and it happens *before* any error
surfaces.

**Why:** caught this only because I ran `pip show`/`.pth` diffing after the
fact — `make dev-install`'s own success message gives no signal that it
landed in the wrong venv.

**How to apply:** every worktree Bash call that touches `pyenv activate`
must chain the two `eval` init lines first. After creating/activating a
worktree venv, verify with `python -c "import sys; print(sys.prefix)"`
before running `make dev-install` — confirm it matches
`~/.pyenv/versions/<worktree-venv-name>`, not the shared `radiologist` venv.
If a bad install already happened, fix it immediately: reactivate the
shared venv properly and re-run `make dev-install` from the **main
checkout** path to restore its `.pth` files before continuing any other
work.
