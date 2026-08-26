---
name: worktree-shell-chaining-blocked
description: worktree-isolated Bash sandbox rejects any multi-command or variable-expansion shell line, not just risky ones
metadata:
  type: feedback
---

In a worktree-isolated agent session, the Bash tool's "too complex to verify path
containment" guard fires on almost any command that chains (`&&`, `;`, multi-line)
or does variable expansion (`$VAR`, `eval "$(...)"`) — even totally benign ones like
`export X=1 && export Y=2` or `echo $VAR`. Only single, flat, literal commands
(`ls`, `cp a b`, `VAR=val /abs/path/to/bin arg`) reliably pass.

**Why:** the guard statically checks that a command cannot possibly leave the
worktree directory; it cannot reason about compound commands or expansions, so it
blocks the whole line rather than risk it.

**How to apply:** in a worktree, never rely on `pyenv activate`/`eval "$(pyenv init -)"`
to set up a venv — shell state doesn't persist between Bash tool calls anyway. Instead,
resolve the target venv's absolute bin path once (e.g. via
`pyenv root`/`ls .../versions/<py>/envs/`) and invoke each tool directly by absolute
path, prefixing a single `VAR=value` env-var assignment inline if needed
(`VIRTUAL_ENV=/abs/path/to/venv uv sync --active ...` works; chaining two `export`s
does not). `make` targets and multi-line here-docs passed to `Write`/`Edit` are fine —
only ad-hoc Bash chaining/expansion is restricted. See [[feedback_pyenv_activate_needs_shell_init]] and [[feedback_worktree_venv_pth_corruption_risk]] for related worktree/venv setup notes.
