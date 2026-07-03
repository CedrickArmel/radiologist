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

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from radiologist.registry.cli import app

runner = CliRunner()


def _make_artifact(qualified_name: str, version: str, aliases=None):
    art = MagicMock()
    art.qualified_name = qualified_name
    art.version = version
    art.aliases = list(aliases or [])
    art.save = MagicMock()
    art.link = MagicMock()
    return art


class TestResolveCommand:
    def test_prints_resolved_qualified_name_and_version(self):
        art = _make_artifact("entity/project/model-R:best", "v3")
        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            result = runner.invoke(app, ["resolve", "entity/project", "--run-id", "R"])

        assert result.exit_code == 0
        assert "entity/project/model-R:best" in result.output
        assert "v3" in result.output

    def test_run_id_and_tags_together_exits_non_zero(self):
        result = runner.invoke(
            app,
            ["resolve", "entity/project", "--run-id", "R", "--tags", "a"],
        )

        assert result.exit_code != 0


class TestPullCommand:
    def test_registry_backed_selector_resolves_then_downloads(self, tmp_path):
        art = _make_artifact("entity/project/model-R:best", "best")
        download_dir = tmp_path / "downloaded"
        download_dir.mkdir()
        ckpt = download_dir / "model.ckpt"
        ckpt.write_bytes(b"ckpt")
        art.download.return_value = str(download_dir)
        local_dir = tmp_path / "out"

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            result = runner.invoke(
                app,
                [
                    "pull",
                    "entity/project",
                    "--local-dir",
                    str(local_dir),
                    "--run-id",
                    "R",
                ],
            )

        assert result.exit_code == 0
        assert str(ckpt) in result.output

    def test_no_selector_flags_treats_positional_as_raw_artifact_path(self, tmp_path):
        art = MagicMock()
        download_dir = tmp_path / "downloaded"
        download_dir.mkdir()
        onnx = download_dir / "model.onnx"
        onnx.write_bytes(b"onnx")
        art.download.return_value = str(download_dir)
        local_dir = tmp_path / "out"

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            result = runner.invoke(
                app,
                ["pull", "entity/project/model:v1", "--local-dir", str(local_dir)],
            )

        assert result.exit_code == 0
        assert str(onnx) in result.output
        art.download.assert_called_once_with(str(local_dir))


class TestPushCommand:
    def test_opens_run_logs_both_artifacts_and_prints_qualified_names(self, tmp_path):
        det_path = tmp_path / "det.onnx"
        mcd_path = tmp_path / "mcd.onnx"
        det_path.write_bytes(b"det")
        mcd_path.write_bytes(b"mcd")
        fake_run = MagicMock()
        fake_run.entity = "entity"
        fake_run.project = "project"

        with (
            patch("radiologist.registry.cli._wandb") as cli_wandb,
            patch("radiologist.registry.uploader._wandb") as uploader_wandb,
        ):
            cli_wandb.init.return_value = fake_run
            uploader_wandb.Artifact.return_value = MagicMock()

            result = runner.invoke(
                app,
                [
                    "push",
                    "--det-path",
                    str(det_path),
                    "--mcd-path",
                    str(mcd_path),
                    "--run-id",
                    "R",
                    "--det-collection",
                    "DC",
                    "--mcd-collection",
                    "MC",
                    "--input-shape",
                    "1",
                    "--input-shape",
                    "1",
                    "--input-shape",
                    "224",
                    "--input-shape",
                    "224",
                    "--classes",
                    "a",
                    "--classes",
                    "b",
                ],
            )

        assert result.exit_code == 0
        cli_wandb.init.assert_called_once_with(job_type="push")
        fake_run.finish.assert_called_once()
        assert "model-R:best" in result.output
        assert "model-R-mcd:best" in result.output


class TestPromoteCommand:
    def test_without_force_declined_confirmation_aborts_and_does_not_promote(self):
        with patch("radiologist.registry.resolver._wandb") as resolver_wandb:
            result = runner.invoke(
                app,
                [
                    "promote",
                    "entity/project",
                    "--run-id",
                    "R",
                    "--det-collection",
                    "DC",
                    "--mcd-collection",
                    "MC",
                ],
                input="n\n",
            )

        assert result.exit_code != 0
        resolver_wandb.Api.assert_not_called()

    def test_with_force_promotes_without_prompting_and_prints_alias(self):
        det_art = _make_artifact("entity/project/model-R:best", "best", ["best"])
        mcd_art = _make_artifact("entity/project/model-R-mcd:best", "best", ["best"])

        with (
            patch("radiologist.registry.resolver._wandb") as resolver_wandb,
            patch("radiologist.registry.collection._wandb") as collection_wandb,
            patch("radiologist.registry.uploader._wandb") as uploader_wandb,
        ):
            resolver_api = MagicMock()
            resolver_wandb.Api.return_value = resolver_api
            resolver_api.artifact.side_effect = [det_art, mcd_art]

            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = []
            collection_api.artifact_collection.return_value = det_collection_obj

            uploader_api = MagicMock()
            uploader_wandb.Api.return_value = uploader_api
            uploader_api.artifact.side_effect = [det_art, mcd_art]

            result = runner.invoke(
                app,
                [
                    "promote",
                    "entity/project",
                    "--run-id",
                    "R",
                    "--det-collection",
                    "DC",
                    "--mcd-collection",
                    "MC",
                    "--force",
                ],
            )

        assert result.exit_code == 0
        assert "production" in result.output


class TestTransitionToProductionCommand:
    def test_force_prints_production_result(self):
        det_staging = _make_artifact("entity/project/model-a:v0", "v0", ["staging"])
        mcd_staging = _make_artifact("entity/project/model-c-mcd:v0", "v0", ["staging"])

        with (
            patch("radiologist.registry.collection._wandb") as collection_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = [det_staging]
            mcd_collection_obj = MagicMock()
            mcd_collection_obj.artifacts.return_value = [mcd_staging]
            collection_api.artifact_collection.side_effect = [
                det_collection_obj,
                mcd_collection_obj,
            ]

            alias_api = MagicMock()
            alias_wandb.Api.return_value = alias_api
            alias_api.artifact.side_effect = [det_staging, mcd_staging]

            result = runner.invoke(
                app,
                [
                    "transition-to-production",
                    "--det-collection",
                    "DC",
                    "--mcd-collection",
                    "MC",
                    "--force",
                ],
            )

        assert result.exit_code == 0
        assert "production" in result.output


class TestListCommand:
    def test_prints_one_line_per_member_with_aliases(self):
        member = _make_artifact("entity/project/model-a:v0", "v0", ["staging"])

        with patch("radiologist.registry.collection._wandb") as collection_wandb:
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            collection_obj = MagicMock()
            collection_obj.artifacts.return_value = [member]
            collection_api.artifact_collection.return_value = collection_obj

            result = runner.invoke(
                app, ["list", "--type", "model", "--collection", "C"]
            )

        assert result.exit_code == 0
        assert "entity/project/model-a:v0" in result.output
        assert "staging" in result.output


class TestAliasCommands:
    def test_set_then_get_reflects_and_remove_removes(self):
        art = _make_artifact("entity/project/model:v1", "v1", [])

        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            set_result = runner.invoke(
                app, ["alias", "set", "entity/project/model:v1", "staging"]
            )
            get_result = runner.invoke(app, ["alias", "get", "entity/project/model:v1"])
            remove_result = runner.invoke(
                app, ["alias", "remove", "entity/project/model:v1", "staging"]
            )

        assert set_result.exit_code == 0
        assert get_result.exit_code == 0
        assert "staging" in get_result.output
        assert remove_result.exit_code == 0
        assert "staging" not in art.aliases


class TestMainEntryPoint:
    def test_raises_runtime_error_when_typer_absent(self):
        import radiologist.registry.cli as cli_mod

        with patch.object(cli_mod, "_typer", None):
            with pytest.raises(RuntimeError, match="cli"):
                cli_mod.main()
