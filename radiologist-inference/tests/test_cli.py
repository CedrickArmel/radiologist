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

"""Tests for the radiologist.inference CLI surface (issue #91).

The skeleton stubs the predict/explain/uncertainty command bodies with
NotImplementedError, so only the command shape is asserted here. Behavioral
CLI tests are owned by the slice issue that implements the smart factory
and CLI (#5).
"""

from unittest.mock import patch

import pytest

from radiologist.inference.cli import app


def _command_names():
    return {cmd.callback.__name__ for cmd in app.registered_commands}


def test_cli_exposes_predict_explain_uncertainty_commands():
    """The CLI must expose exactly predict, explain, and uncertainty."""
    assert _command_names() == {"predict", "explain", "uncertainty"}


def test_cli_no_longer_exposes_pull_command():
    """The pull subcommand must be absent from the CLI."""
    assert "pull" not in _command_names()


def test_cli_entry_point_raises_runtime_error_when_typer_absent():
    """Invoking cli entry point without typer raises RuntimeError naming 'cli' extra."""
    import radiologist.inference.cli as cli_mod

    with patch.object(cli_mod, "_typer", None):
        with pytest.raises(RuntimeError, match="cli"):
            cli_mod.main()
