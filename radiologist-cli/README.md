# radiologist-cli

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-cli)](https://pypi.org/project/radiologist-cli/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

Unified `radiologist` CLI — a single dispatcher fronting the `etl`, `core`,
`registry`, and `infer` command groups. The four business packages
(`radiologist-etl`, `radiologist-core`, `radiologist-registry`,
`radiologist-inference`) are pure libraries; every command body lives here.

## Installation

### Hard dependencies (always installed)

```bash
pip install radiologist-cli
```

### Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `etl` | `radiologist-etl[all]` | `radiologist etl ...` |
| `registry` | `radiologist-registry[all]` | `radiologist registry ...` |
| `inference` | `radiologist-inference[all]` | `radiologist infer ...` |
| `all` | all of the above | every command group |

```bash
pip install "radiologist-cli[all]"
```

## Status

This package currently ships the epic's skeleton: every command is declared
with its full signature and help text, but raises `NotImplementedError`.
Behavior lands in follow-up issues.
