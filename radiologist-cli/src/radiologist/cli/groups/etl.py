# MIT License
#
# Copyright (c) 2026 @CedrickArmel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``radiologist etl`` command group.

Three Hydra-composed subcommands (``extract``, ``assign-split``, ``build``),
one per ETL stage, replacing the retired single-command monolithic ``etl``
entry point: each stage now has its own Hydra configuration root and its own
emitted result record, and can be run, scheduled and re-run independently.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import fsspec  # type: ignore[import-untyped]
import hydra
from omegaconf import DictConfig

from radiologist.cli.main import extract_output_flag
from radiologist.etl import run_assign_split, run_build, run_extract
from radiologist.etl import storage_options_from_cfg as _storage_options_from_cfg
from radiologist.utils.cli import (
    EXIT_ERROR,
    EXIT_OK,
    OUTPUT_ENV_VAR,
    emit,
    exit_code_for,
)

__all__ = [
    "SUBCOMMANDS",
    "extract_main",
    "assign_split_main",
    "build_main",
    "run",
]

SUBCOMMANDS: Tuple[str, ...] = ("extract", "assign-split", "build")


def _usage() -> str:
    return "usage: radiologist etl {" + ",".join(SUBCOMMANDS) + "} ..."


def _ensure_input_exists(
    uri: str, label: str, storage_options: Optional[dict] = None
) -> None:
    """Raise ``FileNotFoundError`` when ``uri`` does not resolve.

    Args:
        uri: Path/URI of the input the stage is about to read.
        label: Human-readable name for the input, used in the error message.
        storage_options: Optional fsspec storage options.

    Raises:
        FileNotFoundError: If the URI does not exist.
    """
    opts = storage_options or {}
    fs, root = fsspec.url_to_fs(uri, **opts)
    if not fs.exists(root):
        raise FileNotFoundError(f"{label} not found: {uri}")


@hydra.main(
    config_path="pkg://radiologist.etl.conf",
    config_name="extract",
    version_base=None,
)
def extract_main(cfg: DictConfig) -> None:
    """Hydra-composed entry point for the ``extract`` subcommand.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.
    """
    try:
        _ensure_input_exists(
            cfg.file_list, "File listing", _storage_options_from_cfg(cfg)
        )
        result = run_extract(cfg)
    except Exception as exc:  # noqa: BLE001 - mapped to an exit code below
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for(exc))

    emit(
        {
            "run_id": result.run_id,
            "manifest_path": result.manifest_path,
            "total": result.total,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "excluded": result.excluded,
        }
    )


@hydra.main(
    config_path="pkg://radiologist.etl.conf",
    config_name="assign_split",
    version_base=None,
)
def assign_split_main(cfg: DictConfig) -> None:
    """Hydra-composed entry point for the ``assign-split`` subcommand.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.
    """
    try:
        _ensure_input_exists(
            cfg.manifests_dir, "Manifests folder", _storage_options_from_cfg(cfg)
        )
        result = run_assign_split(cfg)
    except Exception as exc:  # noqa: BLE001 - mapped to an exit code below
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for(exc))

    emit(
        {
            "run_id": result.run_id,
            "split_manifest_path": result.split_manifest_path,
            "source_manifest_count": result.source_manifest_count,
            "record_count": result.record_count,
            "duplicate_count": result.duplicate_count,
        }
    )


@hydra.main(
    config_path="pkg://radiologist.etl.conf",
    config_name="build",
    version_base=None,
)
def build_main(cfg: DictConfig) -> None:
    """Hydra-composed entry point for the ``build`` subcommand.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.
    """
    try:
        _ensure_input_exists(
            cfg.split_manifest, "Split manifest", _storage_options_from_cfg(cfg)
        )
        result = run_build(cfg)
    except Exception as exc:  # noqa: BLE001 - mapped to an exit code below
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for(exc))

    emit(
        {
            "run_id": result.run_id,
            "output_dir": result.output_dir,
            "manifest_path": result.manifest_path,
            "report_path": result.report_path,
            "shard_count": result.shard_count,
        }
    )


_MAIN_BY_SUBCOMMAND: Dict[str, Callable[[], None]] = {
    "extract": extract_main,
    "assign-split": assign_split_main,
    "build": build_main,
}


def run(argv: List[str]) -> int:
    """Run the ``etl`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``etl``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    # The global output-format flag is extracted from the FULL incoming
    # argv first, before any subcommand parsing, so it is honoured whether
    # it appears before or after the subcommand token.
    remaining, fmt = extract_output_flag(argv)

    if remaining and remaining[0] in ("--help", "-h"):
        print(_usage())
        return EXIT_OK

    if not remaining or remaining[0] not in SUBCOMMANDS:
        print(_usage(), file=sys.stderr)
        return EXIT_ERROR

    subcommand, rest = remaining[0], remaining[1:]
    main_fn = _MAIN_BY_SUBCOMMAND[subcommand]

    sys.argv = [f"radiologist etl {subcommand}", *rest]
    previous_fmt = os.environ.get(OUTPUT_ENV_VAR)
    if fmt is not None:
        os.environ[OUTPUT_ENV_VAR] = fmt
    try:
        main_fn()
    except SystemExit as exc:
        code: Any = exc.code
        if code is None:
            return EXIT_OK
        if isinstance(code, int):
            return code
        return EXIT_ERROR
    finally:
        if fmt is not None:
            if previous_fmt is None:
                os.environ.pop(OUTPUT_ENV_VAR, None)
            else:
                os.environ[OUTPUT_ENV_VAR] = previous_fmt
    return EXIT_OK
