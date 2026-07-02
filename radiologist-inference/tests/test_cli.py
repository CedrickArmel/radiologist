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

"""Tests for the radiologist.inference CLI surface.

Covers the three real subcommands (predict, explain, uncertainty) wired to
Classifier/Explainer/MCDropoutPredictor, the absence of a pull subcommand,
and the typer-absent RuntimeError guard.
"""

from unittest.mock import patch

import numpy as np
import pytest
from _helpers import build_det_onnx, build_mcd_onnx
from PIL import Image as PILImage
from typer.testing import CliRunner

from radiologist.inference.cli import app

runner = CliRunner()


def _command_names():
    return {cmd.callback.__name__ for cmd in app.registered_commands}


def _make_png_path(tmp_path, width: int = 64, height: int = 64) -> str:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    path = tmp_path / "input.png"
    img.save(path, format="PNG")
    return str(path)


def test_cli_exposes_predict_explain_uncertainty_commands():
    """The CLI must expose predict, explain, uncertainty, and serve."""
    assert _command_names() == {"predict", "explain", "uncertainty", "serve"}


def test_cli_no_longer_exposes_pull_command():
    """The pull subcommand must be absent from the CLI."""
    assert "pull" not in _command_names()


def test_cli_pull_invocation_errors_as_unknown_command():
    result = runner.invoke(app, ["pull", "entity/project/model:v1"])
    assert result.exit_code != 0


def test_cli_entry_point_raises_runtime_error_when_typer_absent():
    """Invoking cli entry point without typer raises RuntimeError naming 'cli' extra."""
    import radiologist.inference.cli as cli_mod

    with patch.object(cli_mod, "_typer", None):
        with pytest.raises(RuntimeError, match="cli"):
            cli_mod.main()


class TestPredictCommand:
    def test_predict_exits_0_and_prints_predicted_class(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path, "--model", det_path])

        assert result.exit_code == 0
        assert "Predicted class:" in result.output
        assert "NORMAL" in result.output
        assert "ABNORMAL" in result.output


class TestExplainCommand:
    def test_explain_exits_0_and_prints_predicted_class(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(app, ["explain", image_path, "--model", det_path])

        assert result.exit_code == 0
        assert "Predicted class:" in result.output

    def test_explain_with_out_saves_saliency_map(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)
        out_path = tmp_path / "saliency.npy"

        result = runner.invoke(
            app,
            ["explain", image_path, "--model", det_path, "--out", str(out_path)],
        )

        assert result.exit_code == 0
        assert out_path.exists()
        saved = np.load(out_path)
        assert saved.ndim == 2


class TestUncertaintyCommand:
    def test_uncertainty_exits_0_and_prints_stats(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            [
                "uncertainty",
                image_path,
                "--model",
                det_path,
                "--mcd-model",
                mcd_path,
                "--n-passes",
                "5",
            ],
        )

        assert result.exit_code == 0
        assert "entropy" in result.output.lower()
