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

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from radiologist.registry.wandb_registry import WandbRegistry


def _make_artifact(aliases: List[str]) -> MagicMock:
    art = MagicMock()
    art.aliases = list(aliases)
    art.save = MagicMock()
    return art


class TestGetAliases:
    def test_returns_current_alias_list(self) -> None:
        art = _make_artifact(["staging", "v1"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            result = registry.get_aliases("entity/project/model:v1")

        assert result == ["staging", "v1"]

    def test_returns_copy_not_live_list(self) -> None:
        art = _make_artifact(["staging"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            result = registry.get_aliases("entity/project/model:v1")
            result.append("intruder")

        assert art.aliases == ["staging"]

    def test_does_not_persist(self) -> None:
        art = _make_artifact(["staging"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.get_aliases("entity/project/model:v1")

        art.save.assert_not_called()


class TestSetAlias:
    def test_adds_new_alias_and_persists(self) -> None:
        art = _make_artifact(["staging", "v1"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.set_alias("entity/project/model:v1", "production")

        assert "production" in art.aliases
        art.save.assert_called_once()

    def test_idempotent_when_alias_already_present(self) -> None:
        art = _make_artifact(["staging", "v1"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.set_alias("entity/project/model:v1", "staging")

        assert art.aliases.count("staging") == 1
        art.save.assert_not_called()


class TestRemoveAlias:
    def test_removes_existing_alias_and_persists(self) -> None:
        art = _make_artifact(["staging", "v1"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.remove_alias("entity/project/model:v1", "staging")

        assert "staging" not in art.aliases
        art.save.assert_called_once()

    def test_no_op_when_alias_absent(self) -> None:
        art = _make_artifact(["staging", "v1"])
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.remove_alias("entity/project/model:v1", "nonexistent")

        assert art.aliases == ["staging", "v1"]
        art.save.assert_not_called()


class TestWandbNotInstalled:
    def test_get_aliases_raises_runtime_error_when_wandb_missing(self) -> None:
        with patch("radiologist.registry.optional._wandb", None):
            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="wandb"):
                registry.get_aliases("entity/project/model:v1")

    def test_set_alias_raises_runtime_error_when_wandb_missing(self) -> None:
        with patch("radiologist.registry.optional._wandb", None):
            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="wandb"):
                registry.set_alias("entity/project/model:v1", "production")

    def test_remove_alias_raises_runtime_error_when_wandb_missing(self) -> None:
        with patch("radiologist.registry.optional._wandb", None):
            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="wandb"):
                registry.remove_alias("entity/project/model:v1", "staging")
