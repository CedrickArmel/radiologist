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

"""Tests for the ``radiologist infer serve`` command.

``_uvicorn`` is the process boundary — patched at
``radiologist.inference.optional._uvicorn`` (the owning module), asserting on
the FastAPI app handed to ``run``. ``create_app`` itself is owned code and is
never mocked.
"""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from typer.testing import CliRunner

runner = CliRunner()


def _make_registry_wandb_mock(qualified_name: str = "entity/project/model-run1:best"):
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


class TestServeCommand:
    def test_serve_with_model_source_emits_record_before_accepting_connections(
        self, tmp_path, build_det_onnx, monkeypatch
    ):
        import radiologist.inference.optional as optional_mod
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        det_path = build_det_onnx(tmp_path, filename="det.onnx")

        mock_uvicorn = MagicMock()
        with patch.object(optional_mod, "_uvicorn", mock_uvicorn):
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
        record = json.loads(result.output)
        assert record["host"] == "0.0.0.0"
        assert record["port"] == 9000
        assert record["verb"] == "explain"
        assert record["model_path"] == det_path
        assert record["model_run_id"] is None

    def test_serve_with_registry_run_id_emits_record_with_model_run_id(
        self, tmp_path, build_mcd_onnx, monkeypatch
    ):
        import radiologist.inference.optional as optional_mod
        import radiologist.registry.resolver as resolver_mod
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        mcd_dir = tmp_path / "mcd"
        mcd_dir.mkdir()
        build_mcd_onnx(mcd_dir, filename="mcd.onnx")

        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(mcd_dir)

        mock_uvicorn = MagicMock()
        with patch.object(resolver_mod, "_wandb", mock_wandb):
            with patch.object(optional_mod, "_uvicorn", mock_uvicorn):
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
        record = json.loads(result.output)
        assert record["model_run_id"] == "run1"
        assert record["model_path"] is None
        assert record["verb"] == "uncertainty"

    def test_serve_with_no_model_source_starts_with_no_predictor_and_null_record(
        self, tmp_path, make_png_path, monkeypatch
    ):
        import radiologist.inference.optional as optional_mod
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        image_path = make_png_path(tmp_path)

        mock_uvicorn = MagicMock()
        with patch.object(optional_mod, "_uvicorn", mock_uvicorn):
            result = runner.invoke(app, ["serve"])

        assert result.exit_code == 0, result.output
        mock_uvicorn.run.assert_called_once()
        record = json.loads(result.output)
        assert record["model_path"] is None
        assert record["model_run_id"] is None

        (fastapi_app,), _ = mock_uvicorn.run.call_args
        client = TestClient(fastapi_app)
        with open(image_path, "rb") as f:
            response = client.post(
                "/explain", files={"image": ("i.png", f, "image/png")}
            )
        assert response.status_code == 503

    def test_serve_with_two_verb_flags_exits_nonzero_with_stderr_error(
        self, tmp_path, build_det_onnx, monkeypatch
    ):
        import radiologist.inference.optional as optional_mod
        from radiologist.cli.groups.inference import app

        det_path = build_det_onnx(tmp_path, filename="det.onnx")

        mock_uvicorn = MagicMock()
        with patch.object(optional_mod, "_uvicorn", mock_uvicorn):
            result = runner.invoke(
                app, ["serve", "--predict", "--explain", "--path", det_path]
            )

        assert result.exit_code != 0
        assert result.output.strip() != ""
        mock_uvicorn.run.assert_not_called()

    def test_serve_without_serve_dependency_installed_exits_nonzero_naming_extra(
        self, tmp_path, build_det_onnx
    ):
        import radiologist.inference.optional as optional_mod
        from radiologist.cli.groups.inference import app

        det_path = build_det_onnx(tmp_path, filename="det.onnx")

        with patch.object(optional_mod, "_uvicorn", None):
            result = runner.invoke(app, ["serve", "--path", det_path])

        assert result.exit_code != 0
        assert "serve" in result.output
