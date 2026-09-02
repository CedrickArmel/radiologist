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

## Usage

```bash
radiologist <group> [command] [options]
```

| Group | Fronts | Reference |
|---|---|---|
| `etl` | `radiologist-etl` (three Hydra-composed pipeline stages) | `radiologist etl --help` |
| `core` | `radiologist-core` (Hydra-composed training run) | `radiologist core --help` |
| `registry` | `radiologist-registry` (W&B model registry) | [docs/reference/cli-registry.md](../docs/reference/cli-registry.md) |
| `infer` | `radiologist-inference` (ONNX inference/serving) | [docs/reference/cli-inference.md](../docs/reference/cli-inference.md) |

Every command's final result is a single machine-readable record on stdout —
`key=value` lines by default, or `--output json`/`--output yaml` (`-o` short
form also accepted; global flag, goes before the group name). Errors print
`Error: {message}` on stderr with a non-zero exit code from a shared,
minimal taxonomy (`radiologist.utils.cli.exit_code_for`): `2` when the
referenced artifact/file does not exist, `1` for any other failure.

`etl` fronts three independent subcommands, one per ETL stage --
`radiologist etl extract`, `radiologist etl assign-split`, and
`radiologist etl build` -- each its own Hydra root; `radiologist etl --help`
(no subcommand) lists them, and `radiologist etl <subcommand> --help` prints
that stage's full composed config tree. `core` composes its config with
Hydra directly — `radiologist core --help` prints the full composed config
tree. Both `etl` subcommands and `core` accept `key=value` overrides and
`--multirun` sweeps.
