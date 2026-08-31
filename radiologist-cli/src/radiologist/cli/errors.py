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

"""Repo-wide command error handling.

Single definition replacing the two private ``_exit_on_error`` copies deleted
from ``radiologist-registry/src/radiologist/registry/cli.py`` and
``radiologist-inference/src/radiologist/inference/cli.py`` by the skeleton
issue that introduced this package.
"""

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., None])

__all__ = ["exit_on_error"]


def exit_on_error(func: F) -> F:
    """Wrap a CLI command so an unhandled exception becomes a clean exit.

    Args:
        func: The command function to wrap.

    Returns:
        The wrapped function.
    """
    raise NotImplementedError
