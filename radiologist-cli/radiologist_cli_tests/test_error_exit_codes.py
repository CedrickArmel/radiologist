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

"""Behavioral tests for the shared ``exit_on_error`` command decorator."""

import pytest
import typer


class TestExitOnError:
    def test_file_not_found_error_exits_with_not_found_code(self, capsys) -> None:
        from radiologist.cli.errors import exit_on_error

        @exit_on_error
        def command() -> None:
            raise FileNotFoundError("missing.onnx")

        with pytest.raises(typer.Exit) as excinfo:
            command()

        assert excinfo.value.exit_code == 2
        assert "Error: " in capsys.readouterr().err

    def test_other_exception_exits_with_generic_error_code(self, capsys) -> None:
        from radiologist.cli.errors import exit_on_error

        @exit_on_error
        def command() -> None:
            raise ValueError("bad input")

        with pytest.raises(typer.Exit) as excinfo:
            command()

        captured = capsys.readouterr()
        assert excinfo.value.exit_code == 1
        assert "Error: bad input" in captured.err

    def test_returning_normally_prints_nothing_to_stderr(self, capsys) -> None:
        from radiologist.cli.errors import exit_on_error

        @exit_on_error
        def command() -> str:
            return "ok"

        result = command()

        assert result == "ok"
        assert capsys.readouterr().err == ""
