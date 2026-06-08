# CLAUDE.md

## Project description

Chest X-ray classification.

This project provides a fully reproducible machine-learning pipeline that takes a dataset of labelled chest X-rays and automatically trains a classifier capable of distinguishing abnormal lung findings from healthy lungs.

## Codebase

Stack: PyTorch · Lightning · Hydra · W&B · DVC · Prefect · FastAPI · Streamlit · UV · PyEnv.

### Repository layout

This project adopts a mono-repo layout managed by `UV`.

| Path | Contents |
|---|---|
| `radiologist-app/` | UI |
| `radiologist-core/` | Modeling library — datamodule, VGG-11 backbone, focal loss, training loop |
| `radiologist-etl/` | Data preparation — outlier removal (Haralick GLCM), ImageFolder builder |
| `radiologist-inference/` | Raw and processed X-ray images (tracked by DVC, stored on Google Drive) |
| `radiologist-utils/` | Useful helpers |

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
├── README.md
├── tox.ini
└── uv.lock
```

### uv workspace

Five members: `radiologist-utils`, `radiologist-core`, `radiologist-etl`, `radiologist-inference`, `radiologist-app`.

Each package uses `namespace = true` (no `__init__.py` at the `radiologist/` level). Add new members to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.

Each package sets `module-name = "radiocovid.*"` in its `pyproject.toml` (intentional — do not change). Sources live under `src/radiologist/`; tests import via a `sys.path.insert(0, src/)` shim in each package's `tests/conftest.py`. Mirror this pattern when adding new packages.

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

pyenv activate radiologist && uv sync --active [--extra all] --all-groups  # verify extra all or install all optional individually
pre-commit install                  # install git hooks (required once per clone)
pre-commit install --hook-stage commit-msg
```

**ALWAYS** use the `--active` options for `UV` commands so that `venv` managed locally by PyEnv is used.

Run `uv add --active <package>` to add new package to dependencies.

Run `uv [command] --help` to display `uv`'s help.

Run `pyenv [command] --help` to display `pyenv`'s help.

### Worktrees

Always create worktrees in `PROJECT-ROOT`/.claude/worktrees to maintain clean repo - No exceptions.

[Environment setup](#environment-setup) still apply in worktrees - No exceptions.

When an subagent ends his work in a worktree:

- `cp -r` his memory directory back to `PROJECT-ROOT`/.claude/agent-memory/`agent`.

### Running tests

```bash
uv run --active pytest -q -p no:warnings                          # all packages
uv run --active pytest radiologist-core/tests -q -p no:warnings  # single package
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

Fixtures defined in a `conftest.py` can be used by any test in that package without needing to import them (pytest will automatically discover them).

### Git

- GPG signing is enabled; ensure a `venv` is active with sync'ed project deps so pre-commit tools are on `PATH`.
- Commitizen convention enforced via commitizen's pre-commit hook. Valid prefix types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `revert`, `style`.
- Stacked PRs: branches that depend on other feature branches must target the dependency branch, not `main`. Adjust `gh pr create --base` accordingly.
- After a `pre-commit.ci` remote auto-fix commit, run `git fetch origin <branch> && git rebase origin/<branch>` before the next push to avoid diverged-branch rejection.

## Gotchas

- **[PyTorch]** The sandbox security hook false-positives on the `.eval()` method name. Use `model.train(mode=False)` instead of `model.eval()` in any PyTorch code.
- **[LICENSE]** You MUST NOT add the license header in your code yourself. `pre-commit` we do that.
- **[MEMORY]** Generalise before saving: a gotcha observed on one instance (a class, function, OS, or library) should be written at the level of the broader behavior it exemplifies — not pinned to the specific case that triggered it. When writing a memory or gotcha, ask: is this specific to X, or is X just one case of a wider rule? Write the wider rule; mention X only as an example if it aids clarity.
- **[Extra]** For optional extra deps, write their imports (e.g. `from prefect import ...`) wrapped in `try/except ImportError` with stub no-ops so modules import cleanly without the extra installed.
