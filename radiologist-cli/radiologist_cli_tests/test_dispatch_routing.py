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

"""Behavioral tests for ``run_group`` dispatch and the ``main`` entry point.

Sibling command-group bodies (``groups/etl.py``, ``groups/core.py``,
``groups/registry.py``, ``groups/inference.py``) are stubs owned by other
issues in this epic (#172-#175) and are not implemented here. Per the epic's
own guidance, dispatch mechanics are driven directly: ``run_group`` tests
swap the target group module's public ``run`` function (not yet real, out of
this issue's scope) to observe forwarding; ``main`` tests swap ``run_group``
itself (fully covered by its own dedicated tests below) to isolate main's own
orchestration contract (usage/exit codes/env lifecycle) from dispatch
mechanics.
"""

import os

import pytest


class TestRunGroupDispatch:
    def test_forwards_argv_to_the_mapped_group_module_and_returns_its_code(
        self, monkeypatch
    ) -> None:
        import radiologist.cli.groups.core as core_group
        from radiologist.cli.main import run_group

        captured = {}

        def fake_run(argv):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(core_group, "run", fake_run)

        code = run_group("core", ["train", "epochs=1"])

        assert captured["argv"] == ["train", "epochs=1"]
        assert code == 0

    def test_infer_group_dispatches_to_inference_module(self, monkeypatch) -> None:
        import radiologist.cli.groups.inference as inference_group
        from radiologist.cli.main import run_group

        monkeypatch.setattr(inference_group, "run", lambda argv: 7)

        assert run_group("infer", ["predict", "img.png"]) == 7

    def test_raises_cli_install_hint_when_backing_package_unavailable(
        self, monkeypatch
    ) -> None:
        import radiologist.cli.optional as optional
        from radiologist.cli.main import run_group

        monkeypatch.setattr(optional, "_inference", None)

        with pytest.raises(RuntimeError) as excinfo:
            run_group("infer", ["predict", "img.png"])

        assert "pip install 'radiologist-cli[inference]'" in str(excinfo.value)


class TestMain:
    def test_no_arguments_prints_usage_and_exits_error(
        self, monkeypatch, capsys
    ) -> None:
        import sys

        from radiologist.cli.main import GROUPS, main

        monkeypatch.setattr(sys, "argv", ["radiologist"])

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        for group in GROUPS:
            assert group in err

    def test_unrecognized_group_prints_usage_and_exits_error(
        self, monkeypatch, capsys
    ) -> None:
        import sys

        from radiologist.cli.main import GROUPS, main

        monkeypatch.setattr(sys, "argv", ["radiologist", "bogus"])

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        for group in GROUPS:
            assert group in err

    def test_help_flag_exits_ok_and_lists_groups(self, monkeypatch, capsys) -> None:
        import sys

        from radiologist.cli.main import GROUPS, main

        monkeypatch.setattr(sys, "argv", ["radiologist", "--help"])

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        for group in GROUPS:
            assert group in out

    def test_output_env_var_is_visible_to_the_dispatched_group(
        self, monkeypatch
    ) -> None:
        import importlib
        import sys

        main_module = importlib.import_module("radiologist.cli.main")

        monkeypatch.setattr(
            sys, "argv", ["radiologist", "--output=json", "registry", "resolve", "p"]
        )
        monkeypatch.delenv("RADIOLOGIST_OUTPUT", raising=False)

        observed = {}

        def fake_run_group(group, argv):
            observed["value"] = os.environ.get("RADIOLOGIST_OUTPUT")
            observed["group"] = group
            observed["argv"] = argv
            return 0

        monkeypatch.setattr(main_module, "run_group", fake_run_group)

        with pytest.raises(SystemExit):
            main_module.main()

        assert observed["value"] == "json"
        assert observed["group"] == "registry"
        assert observed["argv"] == ["resolve", "p"]

    def test_prior_output_env_var_is_restored_after_dispatch(self, monkeypatch) -> None:
        import importlib
        import sys

        main_module = importlib.import_module("radiologist.cli.main")

        monkeypatch.setattr(
            sys, "argv", ["radiologist", "--output=json", "registry", "resolve", "p"]
        )
        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "kv")
        monkeypatch.setattr(main_module, "run_group", lambda group, argv: 0)

        with pytest.raises(SystemExit):
            main_module.main()

        assert os.environ.get("RADIOLOGIST_OUTPUT") == "kv"

    def test_no_prior_output_env_var_is_left_unset_after_dispatch(
        self, monkeypatch
    ) -> None:
        import importlib
        import sys

        main_module = importlib.import_module("radiologist.cli.main")

        monkeypatch.setattr(
            sys, "argv", ["radiologist", "--output=json", "registry", "resolve", "p"]
        )
        monkeypatch.delenv("RADIOLOGIST_OUTPUT", raising=False)
        monkeypatch.setattr(main_module, "run_group", lambda group, argv: 0)

        with pytest.raises(SystemExit):
            main_module.main()

        assert "RADIOLOGIST_OUTPUT" not in os.environ

    def test_group_exit_code_is_propagated_as_process_exit_code(
        self, monkeypatch
    ) -> None:
        import importlib
        import sys

        main_module = importlib.import_module("radiologist.cli.main")

        monkeypatch.setattr(sys, "argv", ["radiologist", "registry", "resolve", "p"])
        monkeypatch.setattr(main_module, "run_group", lambda group, argv: 3)

        with pytest.raises(SystemExit) as excinfo:
            main_module.main()

        assert excinfo.value.code == 3

    def test_missing_backing_package_exits_error_with_install_hint(
        self, monkeypatch, capsys
    ) -> None:
        import importlib
        import sys

        main_module = importlib.import_module("radiologist.cli.main")

        monkeypatch.setattr(sys, "argv", ["radiologist", "infer", "predict", "img.png"])

        def raising_run_group(group, argv):
            raise RuntimeError("pip install 'radiologist-cli[inference]'")

        monkeypatch.setattr(main_module, "run_group", raising_run_group)

        with pytest.raises(SystemExit) as excinfo:
            main_module.main()

        assert excinfo.value.code == 1
        assert "pip install 'radiologist-cli[inference]'" in capsys.readouterr().err
