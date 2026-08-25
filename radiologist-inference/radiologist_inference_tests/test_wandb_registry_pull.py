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

"""Behavioral tests for inference model-pull via WandbRegistry.pull() (issue #90).

All tests drive through the public API only. wandb SDK is mocked at the
process boundary since it requires external network/auth. Classifier.from_registry
behavioral coverage lives in test_classifier.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from _helpers import build_det_onnx

from radiologist.registry import WandbRegistry


def _make_wandb_mock(onnx_path):
    """Return a wandb mock whose Api().artifact().download() places onnx_path."""
    mock_wandb = MagicMock()

    artifact = MagicMock()
    artifact.download.return_value = os.path.dirname(onnx_path)

    api_instance = MagicMock()
    api_instance.artifact.return_value = artifact
    mock_wandb.Api.return_value = api_instance

    return mock_wandb


class TestWandbRegistryPull:
    def test_pull_returns_path_to_onnx_file(self, tmp_path):
        """WandbRegistry.pull() must return the local path to an .onnx file."""
        import radiologist.registry.resolver as resolver_mod

        onnx_path = build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            reg = WandbRegistry()
            result = reg.pull(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        assert result.endswith(".onnx")

    def test_pull_file_exists_at_returned_path(self, tmp_path):
        """The path returned by WandbRegistry.pull() must point to an existing file."""
        import radiologist.registry.resolver as resolver_mod

        onnx_path = build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            reg = WandbRegistry()
            result = reg.pull(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        assert os.path.isfile(result)

    def test_pull_raises_runtime_error_when_wandb_absent(self, tmp_path):
        """WandbRegistry.pull() raises RuntimeError when wandb is absent."""
        import radiologist.registry.optional as optional_mod

        with patch.object(optional_mod, "_wandb", None):
            with pytest.raises(RuntimeError):
                reg = WandbRegistry()
                reg.pull(
                    artifact_path="entity/project/name:v0",
                    local_dir=str(tmp_path),
                )
