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

"""Tests for the radiologist.inference CLI (issue #82).

All tests use typer.testing.CliRunner and mock at process boundaries
(filesystem, W&B API via pull_model, ONNX loading via Predictor).
"""

from unittest.mock import MagicMock, patch

import pytest


def _get_runner_and_app():
    """Import typer testing utilities and the CLI app."""
    from typer.testing import CliRunner

    from radiologist.inference.cli import app

    return CliRunner(), app


def test_predict_exits_0_on_valid_image_and_model(tmp_path):
    """predict command prints result and exits 0 when image and model are valid."""
    from radiologist.inference import Prediction

    runner, app = _get_runner_and_app()

    fake_image = tmp_path / "chest.jpg"
    fake_image.write_bytes(b"FAKE")
    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"FAKE")

    mock_prediction = Prediction(
        probabilities={"NORMAL": 0.8, "ABNORMAL": 0.2},
        predicted_class="NORMAL",
    )

    with patch("radiologist.inference.cli.Predictor") as MockPredictor:
        instance = MagicMock()
        instance.predict.return_value = mock_prediction
        MockPredictor.from_path.return_value = instance

        result = runner.invoke(
            app,
            ["predict", str(fake_image), "--model", str(fake_model)],
        )

    assert result.exit_code == 0
    assert "NORMAL" in result.output


def test_pull_exits_0_on_valid_artifact(tmp_path):
    """pull command downloads model and exits 0 when artifact is retrievable."""
    runner, app = _get_runner_and_app()

    with patch("radiologist.inference.cli.WandbRegistry") as MockRegistry:
        mock_instance = MagicMock()
        mock_instance.pull.return_value = str(tmp_path / "model.onnx")
        MockRegistry.return_value = mock_instance

        result = runner.invoke(
            app,
            ["pull", "entity/project/name:v1", "--local-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    mock_instance.pull.assert_called_once()


def test_predict_exits_1_on_unreadable_image(tmp_path):
    """predict command exits 1 when the image path is unreadable."""
    runner, app = _get_runner_and_app()

    fake_model = tmp_path / "model.onnx"
    fake_model.write_bytes(b"FAKE")

    with patch("radiologist.inference.cli.Predictor") as MockPredictor:
        MockPredictor.from_path.side_effect = Exception("cannot load model")

        result = runner.invoke(
            app,
            ["predict", "/nonexistent/image.jpg", "--model", str(fake_model)],
        )

    assert result.exit_code == 1


def test_predict_exits_1_on_unreadable_model(tmp_path):
    """predict command exits 1 when the model path is unreadable."""
    runner, app = _get_runner_and_app()

    fake_image = tmp_path / "chest.jpg"
    fake_image.write_bytes(b"FAKE")

    with patch("radiologist.inference.cli.Predictor") as MockPredictor:
        MockPredictor.from_path.side_effect = FileNotFoundError("model not found")

        result = runner.invoke(
            app,
            ["predict", str(fake_image), "--model", "/nonexistent/model.onnx"],
        )

    assert result.exit_code == 1


def test_pull_exits_1_on_unretrievable_artifact(tmp_path):
    """pull command exits 1 when artifact cannot be retrieved."""
    runner, app = _get_runner_and_app()

    with patch("radiologist.inference.cli.WandbRegistry") as MockRegistry:
        mock_instance = MagicMock()
        mock_instance.pull.side_effect = RuntimeError("W&B download failed")
        MockRegistry.return_value = mock_instance

        result = runner.invoke(
            app,
            ["pull", "entity/project/bad:v0", "--local-dir", str(tmp_path)],
        )

    assert result.exit_code == 1


def test_cli_entry_point_raises_runtime_error_when_typer_absent():
    """Invoking cli entry point without typer raises RuntimeError naming 'cli' extra."""
    import radiologist.inference.cli as cli_mod

    with patch.object(cli_mod, "_typer", None):
        with pytest.raises(RuntimeError, match="cli"):
            cli_mod.main()
