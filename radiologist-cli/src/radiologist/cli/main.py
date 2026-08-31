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

"""Dispatcher for the unified ``radiologist`` console script.

Parses the leading ``--output``/``-o`` flag and the command group name off
``sys.argv``, then delegates the remaining arguments to the matching group's
``run(argv)`` entry point.
"""

import importlib
import os
import sys
from typing import List, Optional, Tuple

from radiologist.utils.cli import EXIT_ERROR, EXIT_OK, OUTPUT_ENV_VAR

GROUPS: Tuple[str, ...] = ("etl", "core", "registry", "infer")

_MODULE_BY_GROUP = {
    "etl": "radiologist.cli.groups.etl",
    "core": "radiologist.cli.groups.core",
    "registry": "radiologist.cli.groups.registry",
    "infer": "radiologist.cli.groups.inference",
}

# Group -> radiologist-cli extra name. Groups absent from this mapping (only
# "core") back onto a hard dependency and need no availability guard.
_EXTRA_BY_GROUP = {
    "etl": "etl",
    "registry": "registry",
    "infer": "inference",
}

__all__ = ["GROUPS", "extract_output_flag", "split_group", "run_group", "main"]


def _usage() -> str:
    return "usage: radiologist {" + ",".join(GROUPS) + "} ..."


def extract_output_flag(argv: List[str]) -> Tuple[List[str], Optional[str]]:
    """Pull the global ``--output``/``-o`` flag out of ``argv``.

    Supports ``--output json``, ``-o json`` and ``--output=json`` forms.

    Args:
        argv: Raw command-line arguments (excluding the program name).

    Returns:
        A tuple of (remaining argv with the flag removed, the flag's value or
        ``None`` when not present).
    """
    result = list(argv)
    for index, token in enumerate(result):
        if token in ("--output", "-o") and index + 1 < len(result):
            fmt = result[index + 1]
            del result[index : index + 2]
            return result, fmt
        if token.startswith("--output="):
            fmt = token.split("=", 1)[1]
            del result[index]
            return result, fmt
    return result, None


def split_group(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    """Split the leading command-group token off ``argv``.

    Args:
        argv: Arguments remaining after :func:`extract_output_flag`.

    Returns:
        A tuple of (group name or ``None`` when absent/unrecognized, the
        remaining argv to forward to that group's ``run``).
    """
    if argv and argv[0] in GROUPS:
        return argv[0], argv[1:]
    return None, argv


def run_group(group: str, argv: List[str]) -> int:
    """Dispatch ``argv`` to the named command group.

    Args:
        group: One of :data:`GROUPS`.
        argv: Arguments to forward to the group's ``run(argv)``.

    Returns:
        The process exit code returned by the group.
    """
    extra = _EXTRA_BY_GROUP.get(group)
    if extra is not None:
        from radiologist.cli.optional import require

        require(extra)

    module = importlib.import_module(_MODULE_BY_GROUP[group])
    return module.run(argv)  # type: ignore[no-any-return]


def main() -> None:
    """Entry point for the ``radiologist`` console script."""
    argv, output_format = extract_output_flag(sys.argv[1:])
    group, rest = split_group(argv)

    if group is None:
        if "--help" in argv or "-h" in argv:
            print(_usage())
            raise SystemExit(EXIT_OK)
        print(_usage(), file=sys.stderr)
        raise SystemExit(EXIT_ERROR)

    previous = os.environ.get(OUTPUT_ENV_VAR)
    if output_format is not None:
        os.environ[OUTPUT_ENV_VAR] = output_format

    try:
        code = run_group(group, rest)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR) from exc
    finally:
        if previous is None:
            os.environ.pop(OUTPUT_ENV_VAR, None)
        else:
            os.environ[OUTPUT_ENV_VAR] = previous

    raise SystemExit(code)
