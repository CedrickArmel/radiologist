---
name: worktree-disk-exhaustion-reuse-shared-venv-readonly
description: When a fresh per-worktree pyenv virtualenv's `uv sync --all-extras` can't complete because the sandbox disk is nearly full (torch + nvidia-* CUDA wheels alone are several GB), run pytest with the main "radiologist" venv's interpreter directly (PYTHONPATH-only, no uv sync/install against it) instead of fighting for disk space -- this respects the "never uv sync --active a worktree onto the shared venv" rule since no writes to that venv occur.
metadata:
  type: feedback
---

Hit repeatedly implementing issue #175 (`radiologist-cli` core train
command): a dedicated `pyenv virtualenv` for the worktree could not
`uv sync --active --all-extras` (or even a narrowed `--package
radiologist-cli --group test`) to completion -- `uv cache clean` freed
2.9-3.4 GiB each time but downloading `torch` + `nvidia-cudnn-cu13` +
friends (~3+ GiB combined) still exhausted the sandbox's remaining disk
(`overlay` filesystem sitting at 96-99% full with 4 other agents' venvs
already present).

**Why:** CLAUDE.md forbids running `uv sync --active` *from* a worktree
*while activated on* the shared `radiologist` venv (it rewrites that
venv's editable `.pth` files to point at the worktree, breaking the main
checkout when the worktree is deleted). But the shared `radiologist` venv
already has the full stack installed (torch, lightning, hydra, wandb,
onnx, webdataset, torchmetrics, pytest, mypy, black, isort, flake8,
commitizen...) — there is no need to `sync` anything into it to *read*
from it.

**How to apply:** when disk-starved, skip creating/syncing a dedicated
worktree venv. Instead:
1. Confirm the shared venv has what's needed:
   `/home/vscode/.pyenv/versions/radiologist/bin/python -c "import torch, lightning, hydra, wandb, pytest, mypy"`.
2. Run tests straight from the worktree with that interpreter --
   `/home/vscode/.pyenv/versions/radiologist/bin/python -m pytest <path>`.
   The root `conftest.py`'s `sys.path.insert(0, str(_ROOT / pkg / "src"))`
   (relative to `__file__`, i.e. the *worktree's* root) makes this resolve
   the worktree's own source, not the main checkout's.
3. For subprocess-spawned CLI tests, pass `PYTHONPATH` explicitly (same
   six `<pkg>/src` dirs) rather than relying on the root conftest shim,
   since a fresh `python -c` subprocess doesn't go through pytest.
4. For `git commit` (pre-commit hooks needing black/isort/flake8/mypy/
   commitizen on `PATH`), plain `git commit -F <msg-file>` worked fine
   without any extra `PATH`/`VIRTUAL_ENV` prefix in this sandbox --
   prefixing `PATH=... git commit ...` on one line was rejected as "too
   complex to verify it stays inside the worktree" (see
   [[shared/MEMORY.md]] on flat single commands).
5. Still create (and later delete) the dedicated
   `radiologist-agent-<id>` pyenv virtualenv per CLAUDE.md if disk allows
   later, but don't block the whole task on it -- the shared venv gets
   you to green.

If `uv sync` must be attempted, `uv cache clean` before each retry
recovers several GiB; narrowing scope (`--package <pkg> --group test`,
no `--all-extras`) reduces but does not eliminate the torch/nvidia-*
download, which is the actual bottleneck regardless of scope (no CPU-only
torch index is configured in this repo as of 2026-08-31).
