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

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from radiologist.cli.groups.registry import app, run

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

    def test_run_id_and_tags_together_exits_non_zero_with_stderr_error(self):
        result = runner.invoke(
            app,
            ["resolve", "entity/project", "--run-id", "R", "--tags", "a"],
        )

        assert result.exit_code != 0
        assert "Error:" in result.output

    def test_without_base_path_argument_exits_non_zero_without_contacting_registry(
        self,
    ):
        with patch("radiologist.registry.resolver._wandb") as resolver_wandb:
            result = runner.invoke(app, ["resolve", "--run-id", "R"])

        assert result.exit_code != 0
        resolver_wandb.Api.assert_not_called()


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
        mock_wandb.Api.assert_called_once()

    def test_missing_artifact_file_after_download_exits_not_found(self, tmp_path):
        art = MagicMock()
        download_dir = tmp_path / "downloaded"
        download_dir.mkdir()
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

        assert result.exit_code == 2

    def test_registry_backed_selector_with_blank_path_exits_non_zero(self, tmp_path):
        local_dir = tmp_path / "out"

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            result = runner.invoke(
                app,
                [
                    "pull",
                    "   ",
                    "--local-dir",
                    str(local_dir),
                    "--run-id",
                    "R",
                ],
            )

        assert result.exit_code != 0
        mock_wandb.Api.assert_not_called()


class TestPushCommand:
    def test_opens_run_logs_both_artifacts_and_emits_record(self, tmp_path):
        det_path = tmp_path / "det.onnx"
        mcd_path = tmp_path / "mcd.onnx"
        det_path.write_bytes(b"det")
        mcd_path.write_bytes(b"mcd")
        fake_run = MagicMock()
        fake_run.entity = "entity"
        fake_run.project = "project"

        with (
            patch("radiologist.registry.optional._wandb") as optional_wandb,
            patch("radiologist.registry.uploader._wandb") as uploader_wandb,
        ):
            optional_wandb.init.return_value = fake_run
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
        optional_wandb.init.assert_called_once_with(job_type="push")
        fake_run.finish.assert_called_once()
        assert "model-R:best" in result.output
        assert "model-R-mcd:best" in result.output
        assert "run_id=R" in result.output or "R" in result.output


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
        assert not isinstance(result.exception, NotImplementedError)
        resolver_wandb.Api.assert_not_called()

    def test_with_force_promotes_without_prompting_and_emits_alias_record(self):
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
        assert "model-R:best" in result.output
        assert "model-R-mcd:best" in result.output

    def test_without_base_path_argument_exits_non_zero_without_contacting_registry(
        self,
    ):
        with patch("radiologist.registry.resolver._wandb") as resolver_wandb:
            result = runner.invoke(
                app,
                [
                    "promote",
                    "--run-id",
                    "R",
                    "--det-collection",
                    "DC",
                    "--mcd-collection",
                    "MC",
                    "--force",
                ],
            )

        assert result.exit_code != 0
        resolver_wandb.Api.assert_not_called()


class TestTransitionToProductionCommand:
    def test_force_emits_production_alias_record(self):
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
    def test_emits_one_record_per_member_with_aliases(self):
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

    def test_json_output_produces_one_parseable_object_per_line(self, monkeypatch):
        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        member_a = _make_artifact("entity/project/model-a:v0", "v0", ["staging"])
        member_b = _make_artifact("entity/project/model-b:v0", "v0", ["production"])

        with patch("radiologist.registry.collection._wandb") as collection_wandb:
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            collection_obj = MagicMock()
            collection_obj.artifacts.return_value = [member_a, member_b]
            collection_api.artifact_collection.return_value = collection_obj

            result = runner.invoke(
                app, ["list", "--type", "model", "--collection", "C"]
            )

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert records[0] == {
            "qualified_name": "entity/project/model-a:v0",
            "aliases": ["staging"],
        }
        assert records[1] == {
            "qualified_name": "entity/project/model-b:v0",
            "aliases": ["production"],
        }


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
        assert "staging" in set_result.output
        assert get_result.exit_code == 0
        assert "staging" in get_result.output
        assert remove_result.exit_code == 0
        assert "staging" in remove_result.output
        assert "staging" not in art.aliases


class TestSingleRecordJsonOutput:
    def test_resolve_json_output_produces_one_parseable_object(self, monkeypatch):
        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        art = _make_artifact("entity/project/model-R:best", "v3")
        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            result = runner.invoke(app, ["resolve", "entity/project", "--run-id", "R"])

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record == {
            "qualified_name": "entity/project/model-R:best",
            "version": "v3",
        }


class TestGenericErrorHandling:
    def test_unhandled_failure_exits_one_with_error_prefix_on_stderr(self):
        with patch("radiologist.registry.alias_manager._wandb") as mock_wandb:
            mock_wandb.Api.side_effect = RuntimeError("boom")

            result = runner.invoke(app, ["alias", "get", "entity/project/model:v1"])

        assert result.exit_code == 1
        assert "Error: boom" in result.output


class TestRunEntryPoint:
    def test_run_returns_zero_on_success(self):
        art = _make_artifact("entity/project/model-R:best", "v3")
        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            code = run(["resolve", "entity/project", "--run-id", "R"])

        assert code == 0

    def test_run_returns_nonzero_on_failure(self):
        code = run(["resolve", "entity/project", "--run-id", "R", "--tags", "a"])

        assert code != 0

    def test_run_with_no_subcommand_exits_cleanly_instead_of_raising(self):
        code = run([])

        assert isinstance(code, int)
        assert code != 0

    def test_run_help_exits_zero_without_raising(self):
        code = run(["--help"])

        assert code == 0
