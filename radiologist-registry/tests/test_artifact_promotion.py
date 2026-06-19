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

import radiologist.registry.uploader as uploader_mod
import radiologist.registry.wandb_registry as facade_mod
from radiologist.registry.models import ExportResult, PromoteResult
from radiologist.registry.uploader import _WandbUploader
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


@pytest.fixture()
def mock_wandb():
    m = MagicMock()
    run = MagicMock()
    m.init.return_value = run

    det_art = MagicMock()
    mcd_art = MagicMock()
    det_linked = MagicMock()
    mcd_linked = MagicMock()
    det_linked.qualified_name = "entity/project/model-abc123:staging"
    mcd_linked.qualified_name = "entity/project/model-abc123-mcd:staging"

    m.Artifact.side_effect = [det_art, mcd_art]
    run.link_artifact.side_effect = [det_linked, mcd_linked]

    return m


class TestWandbUploaderPromote:
    def test_promote_returns_correct_qualified_names(self, export_result, mock_wandb):
        with patch.object(uploader_mod, "_wandb", mock_wandb):
            uploader = _WandbUploader()
            result = uploader.promote(export_result, "entity/project", "staging")

        assert result.det_qualified_name == "entity/project/model-abc123:staging"
        assert result.mcd_qualified_name == "entity/project/model-abc123-mcd:staging"

    def test_promote_creates_run_with_correct_job_type(self, export_result, mock_wandb):
        with patch.object(uploader_mod, "_wandb", mock_wandb):
            uploader = _WandbUploader()
            uploader.promote(export_result, "entity/project", "staging")

        mock_wandb.init.assert_called_once_with(job_type="registry-promote")

    def test_promote_creates_det_artifact_with_correct_name(
        self, export_result, mock_wandb
    ):
        with patch.object(uploader_mod, "_wandb", mock_wandb):
            uploader = _WandbUploader()
            uploader.promote(export_result, "entity/project", "staging")

        calls = mock_wandb.Artifact.call_args_list
        assert calls[0][0][0] == "model-abc123"
        assert calls[0][1]["type"] == "model"

    def test_promote_creates_mcd_artifact_with_correct_name(
        self, export_result, mock_wandb
    ):
        with patch.object(uploader_mod, "_wandb", mock_wandb):
            uploader = _WandbUploader()
            uploader.promote(export_result, "entity/project", "staging")

        calls = mock_wandb.Artifact.call_args_list
        assert calls[1][0][0] == "model-abc123-mcd"
        assert calls[1][1]["type"] == "model"

    def test_promote_links_artifacts_with_alias(self, export_result, mock_wandb):
        with patch.object(uploader_mod, "_wandb", mock_wandb):
            uploader = _WandbUploader()
            uploader.promote(export_result, "entity/project", "staging")

        run = mock_wandb.init.return_value
        link_calls = run.link_artifact.call_args_list
        assert link_calls[0][1]["aliases"] == ["staging"]
        assert link_calls[1][1]["aliases"] == ["staging"]

    def test_promote_raises_runtime_error_when_wandb_absent(self, export_result):
        with patch.object(uploader_mod, "_wandb", None):
            uploader = _WandbUploader()
            with pytest.raises(RuntimeError, match="wandb is required"):
                uploader.promote(export_result, "entity/project", "staging")

    def test_promote_returns_promote_result_instance(self, export_result, mock_wandb):
        with patch.object(uploader_mod, "_wandb", mock_wandb):
            uploader = _WandbUploader()
            result = uploader.promote(export_result, "entity/project", "staging")

        assert isinstance(result, PromoteResult)


class TestWandbRegistryPromoteDelegation:
    def test_registry_promote_delegates_to_uploader(self, export_result, mock_wandb):
        with patch.object(facade_mod, "_wandb", mock_wandb, create=True):
            with patch.object(uploader_mod, "_wandb", mock_wandb):
                registry = WandbRegistry()
                result = registry.promote(export_result, "entity/project", "staging")

        assert result.det_qualified_name == "entity/project/model-abc123:staging"
        assert result.mcd_qualified_name == "entity/project/model-abc123-mcd:staging"
