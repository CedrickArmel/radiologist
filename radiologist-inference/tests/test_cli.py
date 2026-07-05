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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from _helpers import build_det_onnx, build_mcd_onnx
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from typer.testing import CliRunner

from radiologist.inference.cli import app

runner = CliRunner()


def _make_registry_wandb_mock(qualified_name: str = "entity/project/model-run1:best"):
    """A wandb mock distinguishing resolve()'s kwargs artifact() call from
    pull()'s positional artifact() call, per _WandbResolver's two call shapes.
    """
    mock_wandb = MagicMock()

    resolved_art = MagicMock()
    resolved_art.qualified_name = qualified_name
    resolved_art.version = "best"

    pulled_art = MagicMock()

    best_run = MagicMock()
    best_run.id = "run1"

    api_instance = MagicMock()
    api_instance.runs.return_value = [best_run]

    def _artifact(*args, **kwargs):
        return resolved_art if kwargs else pulled_art

    api_instance.artifact.side_effect = _artifact
    mock_wandb.Api.return_value = api_instance
    return mock_wandb, pulled_art


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

        result = runner.invoke(app, ["predict", image_path, "--path", det_path])

        assert result.exit_code == 0
        assert "Predicted class:" in result.output
        assert "NORMAL" in result.output
        assert "ABNORMAL" in result.output


class TestPredictCommandNormalizationFlags:
    def test_predict_with_mean_std_input_shape_changes_output(self, tmp_path):
        """--mean/--std/--input-shape must be accepted and change output vs.
        the default (no flags) invocation."""
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        default_result = runner.invoke(app, ["predict", image_path, "--path", det_path])
        normalized_result = runner.invoke(
            app,
            [
                "predict",
                image_path,
                "--path",
                det_path,
                "--mean",
                "128",
                "--std",
                "65",
                "--input-shape",
                "1,3,224,224",
            ],
        )

        assert default_result.exit_code == 0
        assert normalized_result.exit_code == 0, normalized_result.output
        assert default_result.output != normalized_result.output


class TestExplainCommand:
    def test_explain_exits_0_and_prints_predicted_class(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(app, ["explain", image_path, "--path", det_path])

        assert result.exit_code == 0
        assert "Predicted class:" in result.output

    def test_explain_with_out_saves_saliency_map(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)
        out_path = tmp_path / "saliency.npy"

        result = runner.invoke(
            app,
            ["explain", image_path, "--path", det_path, "--out", str(out_path)],
        )

        assert result.exit_code == 0
        assert out_path.exists()
        saved = np.load(out_path)
        assert saved.ndim == 2


class TestExplainCommandNormalizationFlags:
    def test_explain_with_mean_std_accepted(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            [
                "explain",
                image_path,
                "--path",
                det_path,
                "--mean",
                "128",
                "--std",
                "65",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Predicted class:" in result.output


class TestUncertaintyCommand:
    def test_uncertainty_exits_0_and_prints_stats(self, tmp_path):
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            [
                "uncertainty",
                image_path,
                "--path",
                mcd_path,
                "--n-passes",
                "5",
            ],
        )

        assert result.exit_code == 0
        assert "entropy" in result.output.lower()

    def test_uncertainty_exposes_no_mcd_model_option(self, tmp_path):
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            [
                "uncertainty",
                image_path,
                "--path",
                mcd_path,
                "--mcd-model",
                mcd_path,
            ],
        )

        assert result.exit_code != 0
        assert "--mcd-model" in result.output or "no such option" in (
            result.output.lower()
        )


class TestUncertaintyCommandNormalizationFlags:
    def test_uncertainty_with_mean_std_input_shape_accepted(self, tmp_path):
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            [
                "uncertainty",
                image_path,
                "--path",
                mcd_path,
                "--n-passes",
                "5",
                "--mean",
                "128",
                "--std",
                "65",
                "--input-shape",
                "1,3,224,224",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "entropy" in result.output.lower()


class TestRegistrySelectorDispatch:
    def test_predict_with_run_id_resolves_from_registry(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)
        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            result = runner.invoke(
                app,
                [
                    "predict",
                    image_path,
                    "--run-id",
                    "run1",
                    "--local-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Predicted class:" in result.output
        assert det_path is not None

    def test_predict_with_tags_passes_repeatable_list(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)
        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            result = runner.invoke(
                app,
                [
                    "predict",
                    image_path,
                    "--tags",
                    "a",
                    "--tags",
                    "b",
                    "--local-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        api_instance = mock_wandb.Api.return_value
        _, kwargs = api_instance.runs.call_args
        assert kwargs["filters"]["tags"]["$in"] == ["a", "b"]
        assert det_path is not None

    def test_predict_with_run_id_and_mean_std_input_shape_changes_output(
        self, tmp_path
    ):
        """--mean/--std/--input-shape must not be silently dropped when the
        predictor is loaded via a registry selector (--run-id) instead of
        --path (bugfix #139)."""
        build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        def _invoke(extra_args):
            mock_wandb, pulled_art = _make_registry_wandb_mock()
            pulled_art.download.return_value = str(tmp_path)
            with patch.object(resolver_mod, "_wandb", mock_wandb):
                return runner.invoke(
                    app,
                    [
                        "predict",
                        image_path,
                        "--run-id",
                        "run1",
                        "--local-dir",
                        str(tmp_path),
                        *extra_args,
                    ],
                )

        default_result = _invoke([])
        normalized_result = _invoke(
            ["--mean", "128", "--std", "65", "--input-shape", "1,3,224,224"]
        )

        assert default_result.exit_code == 0, default_result.output
        assert normalized_result.exit_code == 0, normalized_result.output
        assert default_result.output != normalized_result.output

    def test_explain_with_run_id_and_mean_std_input_shape_changes_output(
        self, tmp_path
    ):
        """Same as predict: explain must thread --mean/--std/--input-shape
        through a registry-selector load (bugfix #139)."""
        build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        def _invoke(extra_args):
            mock_wandb, pulled_art = _make_registry_wandb_mock()
            pulled_art.download.return_value = str(tmp_path)
            with patch.object(resolver_mod, "_wandb", mock_wandb):
                return runner.invoke(
                    app,
                    [
                        "explain",
                        image_path,
                        "--run-id",
                        "run1",
                        "--local-dir",
                        str(tmp_path),
                        *extra_args,
                    ],
                )

        default_result = _invoke([])
        normalized_result = _invoke(
            ["--mean", "128", "--std", "65", "--input-shape", "1,3,224,224"]
        )

        assert default_result.exit_code == 0, default_result.output
        assert normalized_result.exit_code == 0, normalized_result.output
        assert default_result.output != normalized_result.output

    def test_predict_without_path_or_selector_exits_nonzero(self, tmp_path):
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path])

        assert result.exit_code != 0
        assert "--path" in result.output
        assert "--run-id" in result.output or "selector" in result.output.lower()

    def test_predict_with_run_id_and_tags_exits_nonzero(self, tmp_path):
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            ["predict", image_path, "--run-id", "run1", "--tags", "a"],
        )

        assert result.exit_code != 0

    def test_predict_with_path_and_run_id_exits_nonzero(self, tmp_path):
        """--path and a registry selector must be mutually exclusive: passing
        both must error rather than silently favoring the registry load."""
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = _make_png_path(tmp_path)

        result = runner.invoke(
            app,
            [
                "predict",
                image_path,
                "--path",
                det_path,
                "--run-id",
                "run1",
            ],
        )

        assert result.exit_code != 0
        assert "--path" in result.output
        assert "--run-id" in result.output

    def test_uncertainty_with_run_id_resolves_mcd_model_via_convention(self, tmp_path):
        """uncertainty --run-id must resolve a single model via the
        ``{run_id}-mcd`` convention — exactly one registry pull, no
        deterministic-model pull (bugfix landed in the verb registry, #142)."""
        mcd_dir = tmp_path / "mcd"
        mcd_dir.mkdir()
        build_mcd_onnx(mcd_dir, filename="mcd.onnx")
        image_path = _make_png_path(tmp_path)

        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(mcd_dir)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            result = runner.invoke(
                app,
                [
                    "uncertainty",
                    image_path,
                    "--run-id",
                    "run1",
                    "--local-dir",
                    str(tmp_path),
                    "--n-passes",
                    "5",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "entropy" in result.output.lower()
        assert pulled_art.download.call_count == 1

    def test_predict_with_run_id_fails_naming_inference_extra_when_wandb_absent(
        self, tmp_path
    ):
        image_path = _make_png_path(tmp_path)
        import radiologist.registry.optional as optional_mod

        with patch.object(optional_mod, "_wandb", None):
            result = runner.invoke(app, ["predict", image_path, "--run-id", "run1"])

        assert result.exit_code != 0
        assert "radiologist-inference[registry]" in result.output


class TestServeCommand:
    def test_serve_with_model_invokes_uvicorn_run(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        import radiologist.inference.cli as cli_mod

        mock_uvicorn = MagicMock()
        with patch.object(cli_mod, "_uvicorn", mock_uvicorn):
            result = runner.invoke(
                app,
                [
                    "serve",
                    "--path",
                    det_path,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_uvicorn.run.assert_called_once()
        _, kwargs = mock_uvicorn.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000

    def test_serve_raises_runtime_error_naming_serve_extra_when_uvicorn_absent(
        self, tmp_path
    ):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        import radiologist.inference.cli as cli_mod

        with patch.object(cli_mod, "_uvicorn", None):
            result = runner.invoke(app, ["serve", "--path", det_path])

        assert result.exit_code != 0
        assert "serve" in result.output

    def test_serve_with_predict_flag_serves_a_classifier(self, tmp_path):
        """--predict must wire a Classifier into the served app: /predict
        works and /explain, /uncertainty are absent (mirrors test_app.py's
        TestClassifierApp, driven end-to-end through the serve command)."""
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        import radiologist.inference.cli as cli_mod

        mock_uvicorn = MagicMock()
        with patch.object(cli_mod, "_uvicorn", mock_uvicorn):
            result = runner.invoke(app, ["serve", "--predict", "--path", det_path])

        assert result.exit_code == 0, result.output
        mock_uvicorn.run.assert_called_once()
        (fastapi_app,), _ = mock_uvicorn.run.call_args
        client = TestClient(fastapi_app)
        image_path = _make_png_path(tmp_path)
        with open(image_path, "rb") as f:
            predict_response = client.post(
                "/predict", files={"image": ("input.png", f, "image/png")}
            )
        assert predict_response.status_code == 200
        with open(image_path, "rb") as f:
            explain_response = client.post(
                "/explain", files={"image": ("input.png", f, "image/png")}
            )
        assert explain_response.status_code == 404

    def test_serve_with_uncertainty_and_run_id_resolves_via_mcd_convention(
        self, tmp_path
    ):
        """--uncertainty --run-id must wire an MCDropoutPredictor into the
        served app via the {run_id}-mcd convention: /uncertainty works and
        /predict is absent (mirrors test_app.py's TestMCDropoutApp, driven
        end-to-end through the serve command)."""
        mcd_dir = tmp_path / "mcd"
        mcd_dir.mkdir()
        build_mcd_onnx(mcd_dir, filename="mcd.onnx")

        import radiologist.inference.cli as cli_mod

        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(mcd_dir)

        import radiologist.registry.resolver as resolver_mod

        mock_uvicorn = MagicMock()
        with patch.object(resolver_mod, "_wandb", mock_wandb):
            with patch.object(cli_mod, "_uvicorn", mock_uvicorn):
                result = runner.invoke(
                    app,
                    [
                        "serve",
                        "--uncertainty",
                        "--run-id",
                        "run1",
                        "--local-dir",
                        str(tmp_path),
                    ],
                )

        assert result.exit_code == 0, result.output
        mock_uvicorn.run.assert_called_once()
        assert pulled_art.download.call_count == 1
        (fastapi_app,), _ = mock_uvicorn.run.call_args
        client = TestClient(fastapi_app)
        image_path = _make_png_path(tmp_path)
        with open(image_path, "rb") as f:
            uncertainty_response = client.post(
                "/uncertainty", files={"image": ("input.png", f, "image/png")}
            )
        assert uncertainty_response.status_code == 200
        with open(image_path, "rb") as f:
            predict_response = client.post(
                "/predict", files={"image": ("input.png", f, "image/png")}
            )
        assert predict_response.status_code == 404

    def test_serve_with_two_verb_flags_exits_nonzero(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        import radiologist.inference.cli as cli_mod

        mock_uvicorn = MagicMock()
        with patch.object(cli_mod, "_uvicorn", mock_uvicorn):
            result = runner.invoke(
                app, ["serve", "--predict", "--explain", "--path", det_path]
            )

        assert result.exit_code != 0

    def test_serve_with_no_source_starts_with_no_predictor(self, tmp_path):
        """No --path/registry selector must start the app with no predictor
        injected: every predictor route 503s (mirrors test_app.py's
        TestNoPredictorInjected, driven end-to-end through the serve
        command)."""
        import radiologist.inference.cli as cli_mod

        mock_uvicorn = MagicMock()
        with patch.object(cli_mod, "_uvicorn", mock_uvicorn):
            result = runner.invoke(app, ["serve"])

        assert result.exit_code == 0, result.output
        mock_uvicorn.run.assert_called_once()
        (fastapi_app,), _ = mock_uvicorn.run.call_args
        client = TestClient(fastapi_app)
        image_path = _make_png_path(tmp_path)
        with open(image_path, "rb") as f:
            predict_response = client.post(
                "/predict", files={"image": ("input.png", f, "image/png")}
            )
        assert predict_response.status_code == 503
