# CLAUDE.md

## Project description

Chest X-ray classification.

This project provides a fully reproducible machine-learning pipeline that takes a dataset of labelled chest X-rays and automatically trains a classifier capable of distinguishing abnormal lung findings from healthy lungs.

## Codebase

Stack: PyTorch · Lightning · Hydra · W&B · DVC · Prefect · FastAPI · Streamlit · UV · PyEnv.

### Module READMEs

Each active package has a README that is the primary reference for its responsibilities, public API, configuration, and design decisions. Read the relevant README before exploring source files.

- `README.md` — end-to-end pipeline overview and quick-start
- `radiologist-utils/README.md` — filesystem helpers, logging, ML utilities
- `radiologist-etl/README.md` — ETL pipeline stages, Haralick GLCM, sharding
- `radiologist-core/README.md` — training loop, datamodule, callbacks, ONNX registry
- `radiologist-inference/README.md` — ONNX inference, W&B Registry pull, Score-CAM, MC-Dropout, FastAPI serving, CLI

### Repository layout

This project adopts a mono-repo layout managed by `UV`.

| Path | Contents |
|---|---|
| `radiologist-app/` | UI |
| `radiologist-core/` | Modeling library — datamodule, configurable backbone (`module` Hydra group, defaults to ResNet-50), focal loss, training loop |
| `radiologist-etl/` | Data preparation — outlier removal (Haralick GLCM), ImageFolder builder |
| `radiologist-inference/` | ONNX inference & serving — pull models from W&B Registry, serve via ONNX Runtime, FastAPI HTTP server, Typer CLI |
| `radiologist-utils/` | Useful helpers |
| `radiologist-registry/` | W&B model registry — promote, resolve, download ONNX artifacts |

```text
├── data.dvc
├── LICENSE
├── Makefile
├── models
├── mypy.ini
├── pyproject.toml
├── radiologist-app
│   ├── pyproject.toml
│   ├── src
│   │   └── radiologist
│   │       └── app
│   │           ├── __init__.py
│   │           └── ...
│   └── tests/
├── radiologist-core
│   ├── pyproject.toml
│   ├── src
│   │   └── radiologist
│   │       └── core
│   │           ├── __init__.py
│   │           └── ...
│   └── tests/
├── radiologist-etl
│   ├── pyproject.toml
│   ├── src
│   │   └── radiologist
│   │       └── etl
│   │           ├── __init__.py
│   │           └── ...
│   └── tests/
├── radiologist-inference
│   ├── pyproject.toml
│   ├── src
│   │   └── radiologist
│   │       └── inference
│   │           ├── __init__.py
│   │           └── ...
│   └── tests/
├── radiologist-utils
│   ├── pyproject.toml
│   ├── src
│   │   └── radiologist
│   │       └── utils
│   │           ├── __init__.py
│   │           └── ...
│   └── tests/
├── radiologist-registry
├── README.md
├── tox.ini
└── uv.lock
```

### uv workspace

Five active members: `radiologist-utils`, `radiologist-core`, `radiologist-etl`, `radiologist-inference`, `radiologist-registry`. `radiologist-app` is **planned but not yet implemented** — its directory does not exist.

Each package uses `namespace = true` (no `__init__.py` at the `radiologist/` level). Add new members to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.

Each package sets `module-name = "radiologist.*"` in its `pyproject.toml` (intentional — do not change). Sources live under `src/radiologist/`; tests import via a `sys.path.insert(0, src/)` shim in each package's `tests/conftest.py`. Mirror this pattern when adding new packages.

### Requirements

- Python 3.10 (pinned via `.python-version`). All generated code must be 3.10-compatible — use `X | Y` unions only where `from __future__ import annotations` is present; otherwise use `Optional[X]`, `Union[X, Y]`, `List[X]`, etc. from `typing`.

### Environment setup

Check if the virtual env defined in `.python-version` exists running `pyenv versions`. If yes proceed to environment setup.

If not create it using:

```bash
pyenv virtualenv 3.10.16 radiologist
```

Setup the envionment running the following:

```bash
pyenv activate radiologist
make dev-install   # sync all deps + extras + install pre-commit hooks
```

**ALWAYS** use the `--active` options for `UV` commands so that `venv` managed locally by PyEnv is used.

Run `uv add --active <package>` to add new package to dependencies.

Run `uv [command] --help` to display `uv`'s help.

Run `pyenv [command] --help` to display `pyenv`'s help.

### Worktrees

Always create worktrees in `PROJECT-ROOT`/.claude/worktrees to maintain clean repo. Never nest worktrees - No exceptions.

**Never share the `radiologist` venv with a worktree.** Editable installs (`uv sync`) write `.pth` files that point at absolute paths. Running `uv sync --active` inside a worktree while activated on the shared `radiologist` venv rewrites those `.pth` files to point at the worktree's path — when the worktree is later deleted, the main checkout's venv silently breaks (imports resolve to a path that no longer exists).

Instead, each worktree gets its own dedicated pyenv virtualenv, named after the worktree:

```bash
pyenv virtualenv 3.10.16 radiologist-<worktree-name>
pyenv activate radiologist-<worktree-name>   # do not rely on .python-version here — activate explicitly
make dev-install
```

[Environment setup](#environment-setup) otherwise still applies (same Python version, same `make dev-install` target) — only the venv name changes.

At the end of the task, before removing the worktree:

```bash
pyenv deactivate
pyenv virtualenv-delete -f radiologist-<worktree-name>
```

Do not leave orphaned `radiologist-<worktree-name>` venvs behind — delete the venv in the same step as the worktree.

When an subagent ends his work in a worktree:

- `cp -r` his memory directory back to `PROJECT-ROOT`/.claude/agent-memory/`agent`.

### Running tests

```bash
make test            # all packages
make test-core       # radiologist-core only
make test-etl        # radiologist-etl only
make test-utils      # radiologist-utils only
make test-inference  # radiologist-inference only
make test-registry   # radiologist-registry only
```

### Code style — PEP 8

PEP 8 is the enforced style guideline. Pre-commit hooks run automatically on `git commit`:

`insert-license` · `isort` · `black` · `flake8` · `mypy` · `Commitizen`

Common fixes needed before a clean commit:

- Remove unused imports (flake8 F401); for untyped third-party libs add `# type: ignore[import-untyped]`
- Use `Optional[T] = None` not `T = None` for optional parameters (mypy `no_implicit_optional`)
- Keep line length ≤ 88 (black default)

### Public APIs

Import public APIs in the `__init__.py` of their module and add them to `__all__` list. Take this into account accessing to them for testing.

### Use `conftest.py` to share fixtures

Fixtures defined in a `conftest.py` can be used by any test in that package without needing to import them (pytest will automatically discover them). A root-level `conftest.py` handles the `sys.path` shim for all 5 packages — do not re-add it per-package.

### Testing philosophy — classist outside-in TDD

**Never mock owned code** (`radiologist.*` or any locally importable module). Only mock true process boundaries: W&B SDK, HTTP, OS/network, external filesystem (fsspec), clock.

- **Lightning components** (`DataModule`, `LightningModule`, `Trainer`) always use real instances — no `MagicMock`, no `SimpleNamespace`. A real `pl.Trainer(fast_dev_run=True, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)` + `trainer.fit(lm, datamodule=dm)` is the test pattern; assert on side-effects after the real fit loop fires hooks.
- **`LModule.load_from_checkpoint`** — use the `ckpt_path` fixture in `radiologist-core/tests/conftest.py` instead of patching. It saves a real Lightning checkpoint to `tmp_path`.
- **Tests drive through public APIs** exported from each package's `__init__.py`, not internal submodules. Patch targets may reference internal paths (e.g. `resolver._wandb`) but the object under test must be the public type.
- **W&B sentinel pattern** — each submodule that calls the W&B SDK exposes a `_wandb` sentinel (e.g. `radiologist.registry.resolver._wandb`). Patch that sentinel, not the whole class that uses it.

### Git

- GPG signing is enabled; ensure a `venv` is active with sync'ed project deps so pre-commit tools are on `PATH`.
- Commitizen convention enforced via commitizen's pre-commit hook. Valid prefix types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `revert`, `style`.
- Stacked PRs: branches that depend on other feature branches must target the dependency branch, not `main`. Adjust `gh pr create --base` accordingly.
- After a `pre-commit.ci` remote auto-fix commit, run `git fetch origin <branch> && git rebase origin/<branch>` before the next push to avoid diverged-branch rejection.
- Keep git tree linear.
- **Mandatory merge type: rebase merge.** Never use a regular merge commit (`--no-ff`) or squash merge — always rebase the branch onto its target first, then fast-forward merge (`git merge --ff-only`) or use `gh pr merge --rebase`. This applies everywhere: worktrees, `main`, and any other branch.

## Gotchas

- **[Packages]** `radiologist-app/` does not exist on disk — it is a planned package. Do not attempt to read or import from it.
- **[LICENSE]** You MUST NOT add the license header in your code yourself. `pre-commit` we do that.
- **[MEMORY]** Generalise before saving: a gotcha observed on one instance (a class, function, OS, or library) should be written at the level of the broader behavior it exemplifies — not pinned to the specific case that triggered it. When writing a memory or gotcha, ask: is this specific to X, or is X just one case of a wider rule? Write the wider rule; mention X only as an example if it aids clarity.
- **[Extra]** For optional extra deps, write their imports (e.g. `from prefect import ...`) wrapped in `try/except ImportError` with stub no-ops so modules import cleanly without the extra installed.
- **[Context]** Large output floods the context window and wastes tokens. Keep Bash output small: pipe through `grep`/`head`/`tail` to extract only what matters, or redirect to `$TMPDIR/out.txt` and read it selectively with the Read tool (`offset`/`limit`). Never `cat` a file via Bash — use the Read tool directly. Never dump raw `find`, `tree`, or long test output into context without filtering.
