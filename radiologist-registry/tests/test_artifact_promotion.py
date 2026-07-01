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

from unittest.mock import MagicMock

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


class TestUploaderLinkToCollectionStubContract:
    def test_link_to_collection_raises_not_implemented(self):
        from radiologist.registry.uploader import _WandbUploader

        with pytest.raises(NotImplementedError):
            _WandbUploader().link_to_collection(
                "entity/project/model-abc123:best",
                "entity/project/model-abc123-mcd:best",
                "det-collection",
                "mcd-collection",
                "staging",
            )


class TestWandbRegistryPromoteStubContract:
    def test_promote_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            WandbRegistry().promote(
                "entity/project", "abc123", "det-collection", "mcd-collection"
            )

    def test_transition_to_production_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            WandbRegistry().transition_to_production("det-collection", "mcd-collection")
