# CLAUDE.md

## Project description

Chest X-ray classification.

This project provides a fully reproducible machine-learning pipeline that takes a dataset of labelled chest X-rays and automatically trains a classifier capable of distinguishing abnormal lung findings from healthy lungs.

## Codebase

Stack: PyTorch · Lightning · Hydra · W&B · DVC · Prefect · FastAPI · Streamlit.

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

### Requirements

- Python 3.10 (pinned via `.python-version`). All generated code must be 3.10-compatible — use `X | Y` unions only where `from __future__ import annotations` is present; otherwise use `Optional[X]`, `Union[X, Y]`, `List[X]`, etc. from `typing`.

### Environment setup

```bash
pyenv activate radiologist
uv sync --active [--extra all] --all-groups  # verify extra all or install all optional individually
pre-commit install                  # install git hooks (required once per clone)
```

### Running tests

```bash
uv run pytest -q                          # all packages
uv run pytest radiologist-core/tests -q   # single package
```

### Code style — PEP 8

PEP 8 is the enforced style guideline. Pre-commit hooks run automatically on `git commit`:

`insert-license` · `isort` · `black` · `flake8` · `mypy`

Common fixes needed before a clean commit:

- Remove unused imports (flake8 F401); for untyped third-party libs add `# type: ignore[import-untyped]`
- Use `Optional[T] = None` not `T = None` for optional parameters (mypy `no_implicit_optional`)
- Keep line length ≤ 88 (black default)

### Git

- GPG signing is enabled; ensure `.venv` is active so pre-commit tools are on `PATH`.
- Commitizen convention enforced (`cz check`). Valid prefix types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `revert`, `style`.

## Gotchas

- **[PyTorch]** The sandbox security hook false-positives on the `.eval()` method name. Use `model.train(mode=False)` instead of `model.eval()` in any PyTorch code.
- **[LICENSE]** No need to add the license header in your code yourself. `pre-commit` we do that.
