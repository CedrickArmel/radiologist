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

"""Behavioral tests for the pure argv-transformation helpers in ``main.py``."""


class TestExtractOutputFlag:
    def test_removes_equals_form_and_captures_format(self) -> None:
        from radiologist.cli.main import extract_output_flag

        argv, fmt = extract_output_flag(["--output=json", "registry", "resolve", "p"])

        assert argv == ["registry", "resolve", "p"]
        assert fmt == "json"

    def test_removes_two_token_form_and_captures_format(self) -> None:
        from radiologist.cli.main import extract_output_flag

        argv, fmt = extract_output_flag(
            ["--output", "json", "registry", "resolve", "p"]
        )

        assert argv == ["registry", "resolve", "p"]
        assert fmt == "json"

    def test_removes_short_flag_form_and_captures_format(self) -> None:
        from radiologist.cli.main import extract_output_flag

        argv, fmt = extract_output_flag(["-o", "json", "registry", "resolve", "p"])

        assert argv == ["registry", "resolve", "p"]
        assert fmt == "json"

    def test_leaves_argv_unchanged_and_returns_none_when_absent(self) -> None:
        from radiologist.cli.main import extract_output_flag

        original = ["registry", "resolve", "p"]
        argv, fmt = extract_output_flag(list(original))

        assert argv == original
        assert fmt is None

    def test_returns_empty_and_none_for_empty_argv(self) -> None:
        from radiologist.cli.main import extract_output_flag

        argv, fmt = extract_output_flag([])

        assert argv == []
        assert fmt is None


class TestSplitGroup:
    def test_splits_known_group_from_forwarded_arguments(self) -> None:
        from radiologist.cli.main import split_group

        group, rest = split_group(["registry", "resolve", "p"])

        assert group == "registry"
        assert rest == ["resolve", "p"]

    def test_returns_none_and_unchanged_argv_for_unknown_leading_token(self) -> None:
        from radiologist.cli.main import split_group

        original = ["bogus", "resolve", "p"]
        group, rest = split_group(list(original))

        assert group is None
        assert rest == original

    def test_returns_none_and_empty_argv_for_empty_input(self) -> None:
        from radiologist.cli.main import split_group

        group, rest = split_group([])

        assert group is None
        assert rest == []
