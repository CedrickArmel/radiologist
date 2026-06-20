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

"""Tests for the radiologist.inference CLI.

All tests use typer.testing.CliRunner and drive real Predictor and
WandbRegistry instances. Only the W&B SDK boundary (_wandb sentinel) is
mocked, and no radiologist.* class is mocked.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.cli import app


def _make_wandb_mock(onnx_path):
    mock_wandb = MagicMock()
    artifact = MagicMock()
    artifact.download.return_value = os.path.dirname(onnx_path)
    api_instance = MagicMock()
    api_instance.artifact.return_value = artifact
    mock_wandb.Api.return_value = api_instance
    return mock_wandb


def _save_png(tmp_path, filename="chest.png"):
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    img_path = str(tmp_path / filename)
    PILImage.fromarray(img_arr).save(img_path)
    return img_path


def _runner():
    from typer.testing import CliRunner

    return CliRunner()


def test_predict_exits_0_on_valid_image_and_model(det_onnx_path, tmp_path):
    """predict command exits 0 and prints the class when given a real model and image."""
    img_path = _save_png(tmp_path)

    result = _runner().invoke(
        app,
        ["predict", img_path, "--model", det_onnx_path],
    )

    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_predict_exits_1_when_model_path_does_not_exist(tmp_path):
    """predict command exits 1 when the model file does not exist."""
    img_path = _save_png(tmp_path)

    result = _runner().invoke(
        app,
        ["predict", img_path, "--model", str(tmp_path / "nonexistent.onnx")],
    )

    assert result.exit_code == 1


def test_predict_exits_1_when_image_path_does_not_exist(det_onnx_path, tmp_path):
    """predict command exits 1 when the image file does not exist."""
    result = _runner().invoke(
        app,
        [
            "predict",
            str(tmp_path / "nonexistent_image.jpg"),
            "--model",
            det_onnx_path,
        ],
    )

    assert result.exit_code == 1


def test_pull_exits_0_on_valid_artifact(det_onnx_path, tmp_path):
    """pull command exits 0 when artifact is retrievable via real WandbRegistry."""
    import radiologist.registry.resolver as resolver_mod

    mock_wandb = _make_wandb_mock(det_onnx_path)

    with patch.object(resolver_mod, "_wandb", mock_wandb):
        result = _runner().invoke(
            app,
            ["pull", "entity/project/name:v1", "--local-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    assert "Model downloaded to:" in result.output


def test_pull_exits_1_when_wandb_sdk_is_absent(tmp_path):
    """pull command exits 1 when the W&B SDK is absent (real registry, _wandb=None)."""
    import radiologist.registry.optional as optional_mod

    with patch.object(optional_mod, "_wandb", None):
        result = _runner().invoke(
            app,
            ["pull", "entity/project/name:v0", "--local-dir", str(tmp_path)],
        )

    assert result.exit_code == 1


def test_cli_entry_point_raises_runtime_error_when_typer_absent():
    """Invoking cli entry point without typer raises RuntimeError naming 'cli' extra."""
    import radiologist.inference.cli as cli_mod

    with patch.object(cli_mod, "_typer", None):
        with pytest.raises(RuntimeError, match="cli"):
            cli_mod.main()
