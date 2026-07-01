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


class TestUploaderStubContract:
    def test_log_model_artifacts_raises_not_implemented(self, export_result):
        from radiologist.registry.uploader import _WandbUploader

        with pytest.raises(NotImplementedError):
            _WandbUploader().log_model_artifacts(
                export_result, MagicMock(), "best.ckpt"
            )

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
