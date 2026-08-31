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

import functools
import sys
from typing import Any, Callable, List, TypeVar

import typer

from radiologist.utils.cli import exit_code_for

F = TypeVar("F", bound=Callable[..., None])

__all__ = ["exit_on_error", "run_typer_app"]


def exit_on_error(func: F) -> F:
    """Wrap a CLI command so an unhandled exception becomes a clean exit.

    Args:
        func: The command function to wrap.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(code=exit_code_for(exc)) from exc

    return wrapper  # type: ignore[return-value]


def run_typer_app(app: typer.Typer, argv: List[str], prog_name: str) -> int:
    """Run a Typer app's command dispatch, translating the outcome into an exit code.

    Individual command bodies are expected to already be wrapped with
    :func:`exit_on_error`, so this only needs to handle what happens
    *outside* a command body: no/unknown subcommand, ``--help``, and a
    user-declined confirmation (``typer.Abort``).

    Args:
        app: The Typer application to run.
        argv: Arguments to dispatch, excluding the program name.
        prog_name: Program name shown in the app's own usage/help output.

    Returns:
        The process exit code.
    """
    from typer.main import get_command

    command = get_command(app)
    try:
        exit_code = command.main(args=argv, prog_name=prog_name, standalone_mode=False)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        # Click/Typer's UsageError/Abort/ClickException family — matched
        # structurally rather than by type since typer vendors its own
        # click fork (``typer._click``) distinct from the ``click`` package.
        show = getattr(exc, "show", None)
        if callable(show):
            show()
            return getattr(exc, "exit_code", 1)
        print(f"Error: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    return exit_code if isinstance(exit_code, int) else 0
