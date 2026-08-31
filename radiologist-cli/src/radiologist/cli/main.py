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

from typing import List, Optional, Tuple

GROUPS: Tuple[str, ...] = ("etl", "core", "registry", "infer")

__all__ = ["GROUPS", "extract_output_flag", "split_group", "run_group", "main"]


def extract_output_flag(argv: List[str]) -> Tuple[List[str], Optional[str]]:
    """Pull the global ``--output``/``-o`` flag out of ``argv``.

    Args:
        argv: Raw command-line arguments (excluding the program name).

    Returns:
        A tuple of (remaining argv with the flag removed, the flag's value or
        ``None`` when not present).
    """
    raise NotImplementedError


def split_group(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    """Split the leading command-group token off ``argv``.

    Args:
        argv: Arguments remaining after :func:`extract_output_flag`.

    Returns:
        A tuple of (group name or ``None`` when absent/unrecognized, the
        remaining argv to forward to that group's ``run``).
    """
    raise NotImplementedError


def run_group(group: str, argv: List[str]) -> int:
    """Dispatch ``argv`` to the named command group.

    Args:
        group: One of :data:`GROUPS`.
        argv: Arguments to forward to the group's ``run(argv)``.

    Returns:
        The process exit code returned by the group.
    """
    raise NotImplementedError


def main() -> None:
    """Entry point for the ``radiologist`` console script."""
    raise NotImplementedError
