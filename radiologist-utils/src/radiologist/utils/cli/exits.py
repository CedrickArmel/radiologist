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

"""Process exit-code taxonomy shared by every ``radiologist`` CLI command.

The mapping is deliberately driven by the exception type callers already
raise (e.g. ``FileNotFoundError`` for a missing artifact/model/image) —
no new exception class is introduced and no existing ``raise`` site changes.
"""

EXIT_OK: int = 0
EXIT_ERROR: int = 1
EXIT_NOT_FOUND: int = 2

__all__ = [
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_NOT_FOUND",
    "exit_code_for",
]


def exit_code_for(exc: BaseException) -> int:
    """Map an exception to a process exit code.

    Args:
        exc: The exception raised by a command.

    Returns:
        :data:`EXIT_NOT_FOUND` for ``FileNotFoundError``, :data:`EXIT_ERROR`
        for any other exception.
    """
    if isinstance(exc, FileNotFoundError):
        return EXIT_NOT_FOUND
    return EXIT_ERROR
