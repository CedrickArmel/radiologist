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

"""Tests for the ``radiologist infer`` predict/explain/uncertainty commands.

Covers keyed-record emission for the three non-serving verbs. ``serve`` has
its own test module (``test_serve_command.py``). The group's commands don't
declare their own ``--output`` flag — the dispatcher (owned by #176) resolves
the global ``--output``/``-o`` flag to the ``RADIOLOGIST_OUTPUT`` env var
before calling a group's ``run(argv)``, so these tests drive the same
resolution path directly via the env var, per
``radiologist.utils.cli.resolve_format``.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
from typer.testing import CliRunner

runner = CliRunner()


def _make_registry_wandb_mock(qualified_name: str = "entity/project/model-run1:best"):
    """A wandb mock covering resolve()'s kwargs-style artifact() call.

    pull() no longer issues its own artifact() call when it immediately
    follows a resolve() of the same qualified name (see _WandbResolver's
    resolved-artifact cache) — the resolved artifact is reused for the
    download, so no separate "pulled" mock is needed.
    """
    mock_wandb = MagicMock()

    resolved_art = MagicMock()
    resolved_art.qualified_name = qualified_name
    resolved_art.version = "best"

    best_run = MagicMock()
    best_run.id = "run1"

    api_instance = MagicMock()
    api_instance.runs.return_value = [best_run]
    api_instance.artifact.return_value = resolved_art
    mock_wandb.Api.return_value = api_instance
    return mock_wandb, resolved_art, api_instance


class TestPredictCommand:
    def test_predict_with_local_path_emits_predicted_class_and_probabilities(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path, "--path", det_path])

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["predicted_class"] in {"NORMAL", "ABNORMAL"}
        assert set(record["probabilities"].keys()) == {"NORMAL", "ABNORMAL"}
        assert all(isinstance(v, float) for v in record["probabilities"].values())

    def test_predict_with_registry_selector_downloads_then_emits_same_record_shape(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)
        mock_wandb, resolved_art, api_instance = _make_registry_wandb_mock()
        resolved_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            result = runner.invoke(
                app,
                [
                    "predict",
                    image_path,
                    "--path",
                    "entity/project",
                    "--run-id",
                    "run1",
                    "--local-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert "predicted_class" in record
        assert set(record["probabilities"].keys()) == {"NORMAL", "ABNORMAL"}
        assert resolved_art.download.call_count == 1
        assert api_instance.artifact.call_count == 1
        assert record["model_qualified_name"] == "entity/project/model-run1:best"
        assert record["model_version"] == "best"

    def test_predict_missing_model_file_exits_2(self, tmp_path, make_png_path):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)
        missing_path = str(tmp_path / "does_not_exist.onnx")

        result = runner.invoke(app, ["predict", image_path, "--path", missing_path])

        assert result.exit_code == 2, result.output

    def test_predict_with_local_path_emits_null_provenance_entries(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path, "--path", det_path])

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["model_qualified_name"] is None
        assert record["model_version"] is None

    def test_predict_with_registry_selector_and_no_path_exits_nonzero(
        self, tmp_path, make_png_path
    ):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path, "--run-id", "run1"])

        assert result.exit_code != 0
        assert "--path" in result.output

    def test_predict_with_neither_path_nor_selector_exits_nonzero(
        self, tmp_path, make_png_path
    ):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path])

        assert result.exit_code != 0


class TestExplainCommand:
    def test_explain_with_out_writes_saliency_map_and_emits_record_with_path(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)
        out_path = tmp_path / "saliency.npy"

        result = runner.invoke(
            app,
            ["explain", image_path, "--path", det_path, "--out", str(out_path)],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        saved = np.load(out_path)
        record = json.loads(result.output)
        assert record["saliency_path"] == str(out_path)
        assert record["saliency_shape"] == list(saved.shape)
        assert "predicted_class" in record

    def test_explain_without_out_emits_null_path_and_writes_no_file(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["explain", image_path, "--path", det_path])

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["saliency_path"] is None
        assert isinstance(record["saliency_shape"], list)
        assert len(list(tmp_path.glob("*.npy"))) == 0
        assert record["model_qualified_name"] is None
        assert record["model_version"] is None

    def test_explain_with_registry_selector_emits_provenance(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)
        mock_wandb, resolved_art, api_instance = _make_registry_wandb_mock()
        resolved_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            result = runner.invoke(
                app,
                [
                    "explain",
                    image_path,
                    "--path",
                    "entity/project",
                    "--run-id",
                    "run1",
                    "--local-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["model_qualified_name"] == "entity/project/model-run1:best"
        assert record["model_version"] == "best"
        assert resolved_art.download.call_count == 1
        assert api_instance.artifact.call_count == 1

    def test_explain_with_registry_selector_and_no_path_exits_nonzero(
        self, tmp_path, make_png_path
    ):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["explain", image_path, "--run-id", "run1"])

        assert result.exit_code != 0
        assert "--path" in result.output

    def test_explain_with_neither_path_nor_selector_exits_nonzero(
        self, tmp_path, make_png_path
    ):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["explain", image_path])

        assert result.exit_code != 0


class TestUncertaintyCommand:
    def test_uncertainty_emits_record_with_entropy_mean_and_std(
        self, tmp_path, build_mcd_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(
            app,
            ["uncertainty", image_path, "--path", mcd_path, "--n-passes", "5"],
        )

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["n_passes"] == 5
        assert isinstance(record["predictive_entropy"], float)
        assert set(record["mean_probabilities"].keys()) == {"NORMAL", "ABNORMAL"}
        assert set(record["std_probabilities"].keys()) == {"NORMAL", "ABNORMAL"}
        assert "predicted_class" in record

    def test_uncertainty_honours_caller_supplied_n_passes(
        self, tmp_path, build_mcd_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(
            app,
            ["uncertainty", image_path, "--path", mcd_path, "--n-passes", "13"],
        )

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["n_passes"] == 13

    def test_uncertainty_with_local_path_emits_null_provenance_entries(
        self, tmp_path, build_mcd_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["uncertainty", image_path, "--path", mcd_path])

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["model_qualified_name"] is None
        assert record["model_version"] is None

    def test_uncertainty_with_registry_selector_emits_provenance(
        self, tmp_path, build_mcd_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        mcd_dir = tmp_path / "mcd"
        mcd_dir.mkdir()
        build_mcd_onnx(mcd_dir, filename="mcd.onnx")
        image_path = make_png_path(tmp_path)
        mock_wandb, resolved_art, api_instance = _make_registry_wandb_mock()
        resolved_art.download.return_value = str(mcd_dir)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            result = runner.invoke(
                app,
                [
                    "uncertainty",
                    image_path,
                    "--path",
                    "entity/project",
                    "--run-id",
                    "run1",
                    "--local-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        record = json.loads(result.output)
        assert record["model_qualified_name"] == "entity/project/model-run1:best"
        assert record["model_version"] == "best"
        assert resolved_art.download.call_count == 1
        assert api_instance.artifact.call_count == 1

    def test_uncertainty_with_registry_selector_and_no_path_exits_nonzero(
        self, tmp_path, make_png_path
    ):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["uncertainty", image_path, "--run-id", "run1"])

        assert result.exit_code != 0
        assert "--path" in result.output

    def test_uncertainty_with_neither_path_nor_selector_exits_nonzero(
        self, tmp_path, make_png_path
    ):
        from radiologist.cli.groups.inference import app

        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["uncertainty", image_path])

        assert result.exit_code != 0


class TestOutputFormat:
    def test_radiologist_output_json_env_var_produces_one_parseable_json_object(
        self, tmp_path, build_det_onnx, make_png_path, monkeypatch
    ):
        from radiologist.cli.groups.inference import app

        monkeypatch.setenv("RADIOLOGIST_OUTPUT", "json")
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)

        result = runner.invoke(app, ["predict", image_path, "--path", det_path])

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)


class TestRun:
    def test_run_returns_0_on_success(
        self, tmp_path, build_det_onnx, make_png_path, capsys
    ):
        from radiologist.cli.groups.inference import run

        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        image_path = make_png_path(tmp_path)

        exit_code = run(["predict", image_path, "--path", det_path])

        assert exit_code == 0

    def test_run_returns_2_when_model_file_missing(self, tmp_path, make_png_path):
        from radiologist.cli.groups.inference import run

        image_path = make_png_path(tmp_path)
        missing_path = str(tmp_path / "does_not_exist.onnx")

        exit_code = run(["predict", image_path, "--path", missing_path])

        assert exit_code == 2

    def test_run_returns_nonzero_for_unknown_command(self):
        from radiologist.cli.groups.inference import run

        exit_code = run(["not-a-command"])

        assert exit_code != 0
