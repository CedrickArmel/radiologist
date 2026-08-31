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

"""Behavioral tests for ``radiologist.cli.optional.require``.

The business packages themselves (``radiologist.etl``, ``radiologist.registry``)
are hard dependencies of ``radiologist-cli`` and are always importable in this
test environment, so the "package absent" path is exercised by monkeypatching
the module-level sentinels in :mod:`radiologist.cli.optional` -- that module
is the intended seam per the issue's technical notes.
"""

import pytest


class TestRequire:
    def test_returns_the_module_when_available_and_feature_complete(self) -> None:
        import radiologist.etl as etl_module
        from radiologist.cli.optional import require

        assert require("etl") is etl_module

    def test_raises_cli_hint_when_package_is_absent(self, monkeypatch) -> None:
        import radiologist.cli.optional as optional

        monkeypatch.setattr(optional, "_inference", None)

        from radiologist.cli.optional import require

        with pytest.raises(RuntimeError) as excinfo:
            require("inference")

        assert "pip install 'radiologist-cli[inference]'" in str(excinfo.value)

    def test_raises_cli_hint_not_business_hint_when_etl_feature_sentinel_missing(
        self, monkeypatch
    ) -> None:
        from radiologist.cli.optional import require
        from radiologist.etl import prefect_pipelines

        monkeypatch.setattr(prefect_pipelines, "_PREFECT_AVAILABLE", False)

        with pytest.raises(RuntimeError) as excinfo:
            require("etl")

        message = str(excinfo.value)
        assert "pip install 'radiologist-cli[etl]'" in message
        assert "radiologist-etl" not in message

    def test_raises_cli_hint_not_business_hint_when_registry_feature_sentinel_missing(
        self, monkeypatch
    ) -> None:
        from radiologist.cli.optional import require
        from radiologist.registry import optional as registry_optional

        monkeypatch.setattr(registry_optional, "_wandb", None)

        with pytest.raises(RuntimeError) as excinfo:
            require("registry")

        message = str(excinfo.value)
        assert "pip install 'radiologist-cli[registry]'" in message
        assert "radiologist-registry" not in message
