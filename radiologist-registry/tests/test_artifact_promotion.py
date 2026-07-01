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

from radiologist.registry.models import ExportResult
from radiologist.registry.wandb_registry import WandbRegistry


@pytest.fixture()
def export_result(tmp_path):
    det = tmp_path / "model.onnx"
    mcd = tmp_path / "model_mcd.onnx"
    det.write_bytes(b"det")
    mcd.write_bytes(b"mcd")
    return ExportResult(
        det_path=str(det),
        mcd_path=str(mcd),
        run_id="abc123",
        input_shape=(1, 3, 224, 224),
        classes=["normal", "abnormal"],
    )


class TestUploaderLogModelArtifacts:
    def test_logs_det_and_mcd_artifacts_with_best_alias(self, export_result, tmp_path):
        from radiologist.registry.uploader import _WandbUploader

        ckpt = tmp_path / "best.ckpt"
        ckpt.write_bytes(b"ckpt")
        fake_run = MagicMock()

        _WandbUploader().log_model_artifacts(export_result, fake_run, str(ckpt))

        aliases_seen = [
            call.kwargs.get("aliases") for call in fake_run.log_artifact.call_args_list
        ]
        assert aliases_seen.count(["best"]) == 2

        names_logged = [
            call.args[0].name for call in fake_run.log_artifact.call_args_list
        ]
        assert f"model-{export_result.run_id}" in names_logged
        assert f"model-{export_result.run_id}-mcd" in names_logged

    def test_det_artifact_contains_both_onnx_and_checkpoint_files(
        self, export_result, tmp_path
    ):
        from radiologist.registry.uploader import _WandbUploader

        ckpt = tmp_path / "best.ckpt"
        ckpt.write_bytes(b"ckpt")
        fake_run = MagicMock()

        _WandbUploader().log_model_artifacts(export_result, fake_run, str(ckpt))

        det_art = next(
            call.args[0]
            for call in fake_run.log_artifact.call_args_list
            if call.args[0].name == f"model-{export_result.run_id}"
            and call.kwargs.get("aliases") == ["best"]
        )
        manifest_paths = {e.path for e in det_art.manifest.entries.values()}
        assert any(p.endswith(".onnx") for p in manifest_paths)
        assert any(p.endswith(".ckpt") for p in manifest_paths)

    def test_logs_last_alias_version_when_last_checkpoint_given(
        self, export_result, tmp_path
    ):
        from radiologist.registry.uploader import _WandbUploader

        ckpt = tmp_path / "best.ckpt"
        ckpt.write_bytes(b"ckpt")
        last_ckpt = tmp_path / "last.ckpt"
        last_ckpt.write_bytes(b"last")
        fake_run = MagicMock()

        _WandbUploader().log_model_artifacts(
            export_result, fake_run, str(ckpt), str(last_ckpt)
        )

        aliases_seen = [
            call.kwargs.get("aliases") for call in fake_run.log_artifact.call_args_list
        ]
        assert ["last"] in aliases_seen

    def test_no_last_alias_version_when_last_checkpoint_absent(
        self, export_result, tmp_path
    ):
        from radiologist.registry.uploader import _WandbUploader

        ckpt = tmp_path / "best.ckpt"
        ckpt.write_bytes(b"ckpt")
        fake_run = MagicMock()

        _WandbUploader().log_model_artifacts(export_result, fake_run, str(ckpt))

        aliases_seen = [
            call.kwargs.get("aliases") for call in fake_run.log_artifact.call_args_list
        ]
        assert ["last"] not in aliases_seen

    def test_returns_logged_artifacts_carrying_run_id(self, export_result, tmp_path):
        from radiologist.registry.uploader import _WandbUploader

        ckpt = tmp_path / "best.ckpt"
        ckpt.write_bytes(b"ckpt")
        fake_run = MagicMock()

        result = _WandbUploader().log_model_artifacts(
            export_result, fake_run, str(ckpt)
        )

        assert result.run_id == export_result.run_id
        assert f"model-{export_result.run_id}" in result.det_qualified_name
        assert f"model-{export_result.run_id}-mcd" in result.mcd_qualified_name

    def test_does_not_link_artifacts_to_a_collection(self, export_result, tmp_path):
        from radiologist.registry.uploader import _WandbUploader

        ckpt = tmp_path / "best.ckpt"
        ckpt.write_bytes(b"ckpt")
        fake_run = MagicMock()

        _WandbUploader().log_model_artifacts(export_result, fake_run, str(ckpt))

        assert not fake_run.link_artifact.called


def _make_artifact(qualified_name: str, aliases: List[str]) -> MagicMock:
    art = MagicMock()
    art.qualified_name = qualified_name
    art.aliases = list(aliases)
    art.save = MagicMock()
    art.link = MagicMock()
    return art


class TestUploaderLinkToCollection:
    def test_links_det_and_mcd_artifacts_with_given_alias(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
        with patch("radiologist.registry.uploader._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.side_effect = [det_art, mcd_art]

            from radiologist.registry.uploader import _WandbUploader

            result = _WandbUploader().link_to_collection(
                det_art.qualified_name,
                mcd_art.qualified_name,
                "det-collection",
                "mcd-collection",
                "production",
            )

        det_art.link.assert_called_once_with("det-collection", aliases=["production"])
        mcd_art.link.assert_called_once_with("mcd-collection", aliases=["production"])
        assert result.det_qualified_name == det_art.qualified_name
        assert result.mcd_qualified_name == mcd_art.qualified_name
        assert result.alias == "production"


class TestUploaderLinkToCollectionRollback:
    def test_reverts_det_link_alias_when_mcd_link_raises(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
        mcd_art.link.side_effect = RuntimeError("transient network error")
        with (
            patch("radiologist.registry.uploader._wandb") as mock_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.side_effect = [det_art, mcd_art]

            alias_api = MagicMock()
            alias_wandb.Api.return_value = alias_api
            det_art.aliases.append("production")
            alias_api.artifact.return_value = det_art

            from radiologist.registry.uploader import _WandbUploader

            with pytest.raises(RuntimeError, match="transient network error"):
                _WandbUploader().link_to_collection(
                    det_art.qualified_name,
                    mcd_art.qualified_name,
                    "det-collection",
                    "mcd-collection",
                    "production",
                )

        alias_api.artifact.assert_called_once_with(det_art.qualified_name)
        assert "production" not in det_art.aliases
        det_art.save.assert_called_once()

    def test_revert_failure_does_not_swallow_original_exception(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
        mcd_art.link.side_effect = RuntimeError("original failure")
        with (
            patch("radiologist.registry.uploader._wandb") as mock_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.side_effect = [det_art, mcd_art]

            alias_wandb.Api.side_effect = RuntimeError("revert also fails")

            from radiologist.registry.uploader import _WandbUploader

            with pytest.raises(RuntimeError, match="original failure"):
                _WandbUploader().link_to_collection(
                    det_art.qualified_name,
                    mcd_art.qualified_name,
                    "det-collection",
                    "mcd-collection",
                    "production",
                )


class TestWandbRegistryPromote:
    def test_applies_production_when_collection_has_no_production_member(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
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

            registry = WandbRegistry()
            result = registry.promote(
                "entity/project", "abc123", "det-collection", "mcd-collection"
            )

        det_art.link.assert_called_once_with("det-collection", aliases=["production"])
        mcd_art.link.assert_called_once_with("mcd-collection", aliases=["production"])
        assert result.alias == "production"

    def test_applies_staging_when_collection_already_has_production_member(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
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
            existing_production = _make_artifact(
                "entity/project/model-xyz:v0", ["production"]
            )
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = [existing_production]
            collection_api.artifact_collection.return_value = det_collection_obj

            uploader_api = MagicMock()
            uploader_wandb.Api.return_value = uploader_api
            uploader_api.artifact.side_effect = [det_art, mcd_art]

            registry = WandbRegistry()
            result = registry.promote(
                "entity/project", "abc123", "det-collection", "mcd-collection"
            )

        det_art.link.assert_called_once_with("det-collection", aliases=["staging"])
        mcd_art.link.assert_called_once_with("mcd-collection", aliases=["staging"])
        assert result.alias == "staging"

    def test_applies_staging_when_only_mcd_collection_has_production_member(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
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
            existing_production = _make_artifact(
                "entity/project/model-xyz-mcd:v0", ["production"]
            )
            mcd_collection_obj = MagicMock()
            mcd_collection_obj.artifacts.return_value = [existing_production]
            collection_api.artifact_collection.side_effect = [
                det_collection_obj,
                mcd_collection_obj,
            ]

            uploader_api = MagicMock()
            uploader_wandb.Api.return_value = uploader_api
            uploader_api.artifact.side_effect = [det_art, mcd_art]

            registry = WandbRegistry()
            result = registry.promote(
                "entity/project", "abc123", "det-collection", "mcd-collection"
            )

        det_art.link.assert_called_once_with("det-collection", aliases=["staging"])
        mcd_art.link.assert_called_once_with("mcd-collection", aliases=["staging"])
        assert result.alias == "staging"

    def test_does_not_download_any_local_file(self):
        det_art = _make_artifact("entity/project/model-abc123:best", ["best"])
        mcd_art = _make_artifact("entity/project/model-abc123-mcd:best", ["best"])
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

            registry = WandbRegistry()
            registry.promote(
                "entity/project", "abc123", "det-collection", "mcd-collection"
            )

        det_art.download.assert_not_called()
        mcd_art.download.assert_not_called()


class TestWandbRegistryTransitionToProduction:
    def test_flips_staging_to_production_in_both_collections(self):
        det_staging = _make_artifact("entity/project/model-a:v0", ["staging"])
        det_production = _make_artifact("entity/project/model-b:v0", ["production"])
        mcd_staging = _make_artifact("entity/project/model-c-mcd:v0", ["staging"])
        mcd_production = _make_artifact("entity/project/model-d-mcd:v0", ["production"])
        with (
            patch("radiologist.registry.collection._wandb") as collection_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = [det_staging, det_production]
            mcd_collection_obj = MagicMock()
            mcd_collection_obj.artifacts.return_value = [mcd_staging, mcd_production]
            collection_api.artifact_collection.side_effect = [
                det_collection_obj,
                mcd_collection_obj,
            ]

            alias_api = MagicMock()
            alias_wandb.Api.return_value = alias_api
            alias_api.artifact.side_effect = [
                det_production,
                det_staging,
                mcd_production,
                mcd_staging,
            ]

            registry = WandbRegistry()
            result = registry.transition_to_production(
                "det-collection", "mcd-collection"
            )

        assert "production" not in det_production.aliases
        assert "production" in det_staging.aliases
        assert "production" not in mcd_production.aliases
        assert "production" in mcd_staging.aliases
        assert result.alias == "production"
        assert result.det_qualified_name == det_staging.qualified_name
        assert result.mcd_qualified_name == mcd_staging.qualified_name

    def test_raises_lookup_error_when_no_staging_member_in_a_collection(self):
        det_production = _make_artifact("entity/project/model-b:v0", ["production"])
        mcd_staging = _make_artifact("entity/project/model-c-mcd:v0", ["staging"])
        with (
            patch("radiologist.registry.collection._wandb") as collection_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = [det_production]
            mcd_collection_obj = MagicMock()
            mcd_collection_obj.artifacts.return_value = [mcd_staging]
            collection_api.artifact_collection.side_effect = [
                det_collection_obj,
                mcd_collection_obj,
            ]

            registry = WandbRegistry()
            with pytest.raises(LookupError):
                registry.transition_to_production("det-collection", "mcd-collection")

        alias_wandb.Api.assert_not_called()
        assert "production" in det_production.aliases

    def test_reverts_det_alias_state_when_mcd_alias_change_raises(self):
        det_staging = _make_artifact("entity/project/model-a:v0", ["staging"])
        det_production = _make_artifact("entity/project/model-b:v0", ["production"])
        mcd_staging = _make_artifact("entity/project/model-c-mcd:v0", ["staging"])
        mcd_production = _make_artifact("entity/project/model-d-mcd:v0", ["production"])
        with (
            patch("radiologist.registry.collection._wandb") as collection_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = [det_staging, det_production]
            mcd_collection_obj = MagicMock()
            mcd_collection_obj.artifacts.return_value = [mcd_staging, mcd_production]
            collection_api.artifact_collection.side_effect = [
                det_collection_obj,
                mcd_collection_obj,
            ]

            alias_api = MagicMock()
            alias_wandb.Api.return_value = alias_api
            alias_api.artifact.side_effect = [
                det_production,
                det_staging,
                RuntimeError("mcd alias change failed"),
                det_staging,
                det_production,
            ]

            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="mcd alias change failed"):
                registry.transition_to_production("det-collection", "mcd-collection")

        assert "production" in det_production.aliases
        assert "production" not in det_staging.aliases

    def test_revert_failure_does_not_swallow_original_transition_exception(self):
        det_staging = _make_artifact("entity/project/model-a:v0", ["staging"])
        det_production = _make_artifact("entity/project/model-b:v0", ["production"])
        mcd_staging = _make_artifact("entity/project/model-c-mcd:v0", ["staging"])
        mcd_production = _make_artifact("entity/project/model-d-mcd:v0", ["production"])
        with (
            patch("radiologist.registry.collection._wandb") as collection_wandb,
            patch("radiologist.registry.alias_manager._wandb") as alias_wandb,
        ):
            collection_api = MagicMock()
            collection_wandb.Api.return_value = collection_api
            det_collection_obj = MagicMock()
            det_collection_obj.artifacts.return_value = [det_staging, det_production]
            mcd_collection_obj = MagicMock()
            mcd_collection_obj.artifacts.return_value = [mcd_staging, mcd_production]
            collection_api.artifact_collection.side_effect = [
                det_collection_obj,
                mcd_collection_obj,
            ]

            alias_api = MagicMock()
            alias_wandb.Api.return_value = alias_api
            alias_api.artifact.side_effect = [
                det_production,
                det_staging,
                RuntimeError("original mcd failure"),
                RuntimeError("revert also fails"),
            ]

            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="original mcd failure"):
                registry.transition_to_production("det-collection", "mcd-collection")
