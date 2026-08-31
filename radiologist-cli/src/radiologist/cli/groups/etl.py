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

"""``radiologist etl`` command group — Hydra-composed ETL pipeline entry point."""

import os
import sys
from typing import List, Optional, Tuple

import fsspec  # type: ignore[import-untyped]
import hydra
from omegaconf import DictConfig, OmegaConf

from radiologist.etl import etl_flow
from radiologist.utils.cli import OUTPUT_ENV_VAR, emit, exit_code_for

__all__ = ["etl_main", "run"]


def _extract_output_flag(argv: List[str]) -> Tuple[List[str], Optional[str]]:
    """Pull the leading ``--output``/``-o`` flag out of ``argv``.

    Supports ``--output json``, ``-o json`` and ``--output=json`` forms.

    Args:
        argv: Raw arguments forwarded to the ``etl`` group.

    Returns:
        A tuple of (remaining argv with the flag removed, the flag's value
        or ``None`` when not present).
    """
    remaining: List[str] = []
    fmt: Optional[str] = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("--output", "-o") and i + 1 < len(argv):
            fmt = argv[i + 1]
            i += 2
            continue
        if token.startswith("--output="):
            fmt = token.split("=", 1)[1]
            i += 1
            continue
        remaining.append(token)
        i += 1
    return remaining, fmt


def _ensure_source_exists(cfg: DictConfig) -> None:
    """Raise ``FileNotFoundError`` when ``cfg.source`` does not resolve.

    Args:
        cfg: Composed Hydra config; ``cfg.source`` and the optional
            ``cfg.storage_options`` are inspected.

    Raises:
        FileNotFoundError: If the source path/URI does not exist.
    """
    storage_options = OmegaConf.select(cfg, "storage_options")
    _raw = OmegaConf.to_container(storage_options) if storage_options else None
    opts: dict = dict(_raw) if isinstance(_raw, dict) else {}
    fs, root = fsspec.url_to_fs(cfg.source, **opts)
    if not fs.exists(root):
        raise FileNotFoundError(f"ETL source not found: {cfg.source}")


@hydra.main(
    config_path="pkg://radiologist.etl.conf",
    config_name="etl",
    version_base=None,
)
def etl_main(cfg: DictConfig) -> None:
    """Hydra-composed entry point for the ``radiologist etl`` command group.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.
    """
    try:
        _ensure_source_exists(cfg)
        result = etl_flow(cfg)
    except Exception as exc:  # noqa: BLE001 - mapped to an exit code below
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(exit_code_for(exc))

    emit({"run_id": result.run_id, "manifest_path": result.manifest_path})


def run(argv: List[str]) -> int:
    """Run the ``etl`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``etl``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    remaining, fmt = _extract_output_flag(argv)
    sys.argv = ["radiologist etl"] + remaining
    previous_fmt = os.environ.get(OUTPUT_ENV_VAR)
    if fmt is not None:
        os.environ[OUTPUT_ENV_VAR] = fmt
    try:
        etl_main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        if fmt is not None:
            if previous_fmt is None:
                os.environ.pop(OUTPUT_ENV_VAR, None)
            else:
                os.environ[OUTPUT_ENV_VAR] = previous_fmt
    return 0
