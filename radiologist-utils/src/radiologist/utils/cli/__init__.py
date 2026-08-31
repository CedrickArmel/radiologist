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

"""Machine-readable output helpers shared by every ``radiologist`` CLI command.

Constants are real values — they are data, not behavior, so downstream
skeletons can rely on them immediately. ``resolve_format``, ``emit``, and
``exit_code_for`` are stubbed with ``NotImplementedError`` bodies; their
behavior lands in a follow-up issue.
"""

from typing import Any, Mapping, Optional, TextIO, Tuple

OUTPUT_FORMATS: Tuple[str, ...] = ("kv", "json", "yaml")
DEFAULT_OUTPUT_FORMAT: str = "kv"
OUTPUT_ENV_VAR: str = "RADIOLOGIST_OUTPUT"
EXIT_OK: int = 0
EXIT_ERROR: int = 1
EXIT_NOT_FOUND: int = 2

__all__ = [
    "OUTPUT_FORMATS",
    "DEFAULT_OUTPUT_FORMAT",
    "OUTPUT_ENV_VAR",
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_NOT_FOUND",
    "resolve_format",
    "emit",
    "exit_code_for",
]


def resolve_format(explicit: Optional[str] = None) -> str:
    """Resolve the effective output format.

    Args:
        explicit: Format requested via a CLI flag, taking precedence over the
            ``RADIOLOGIST_OUTPUT`` environment variable and the default.

    Returns:
        One of :data:`OUTPUT_FORMATS`.
    """
    raise NotImplementedError


def emit(
    data: Mapping[str, Any],
    fmt: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """Write ``data`` to ``stream`` in the resolved output format.

    Args:
        data: Mapping of result fields to serialize.
        fmt: Explicit output format, resolved via :func:`resolve_format` when
            not given.
        stream: Destination stream, defaults to ``sys.stdout``.
    """
    raise NotImplementedError


def exit_code_for(exc: BaseException) -> int:
    """Map an exception to a process exit code.

    Args:
        exc: The exception raised by a command.

    Returns:
        One of :data:`EXIT_OK`, :data:`EXIT_ERROR`, :data:`EXIT_NOT_FOUND`.
    """
    raise NotImplementedError
