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

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from radiologist.registry.models import ArtifactRef
from radiologist.registry.wandb_registry import WandbRegistry


def _make_artifact(
    qualified_name: str,
    version: str,
    run_id: str,
) -> MagicMock:
    art = MagicMock()
    art.qualified_name = qualified_name
    art.version = version
    name_with_version = qualified_name.split("/")[-1]
    art.name = name_with_version
    source_run = MagicMock()
    source_run.id = run_id
    art.logged_by.return_value = source_run
    return art


def _make_run(run_id: str, score: float) -> MagicMock:
    run = MagicMock()
    run.id = run_id
    run.summary = {"best_val_score": score}
    return run


class TestResolveViaRunId:
    def test_returns_artifactref_with_supplied_run_id(self, tmp_path: Any) -> None:
        art = _make_artifact(
            "entity/project/model-run123:v3",
            "v3",
            "run123",
        )
        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            ref = registry.resolve("entity/project", run_id="run123")

        assert ref.run_id == "run123"
        assert ref.qualified_name == "entity/project/model-run123:v3"
        assert ref.version == "v3"
        assert ref.artifact_name == "model-run123"

    def test_uses_best_alias_when_no_version_given(self, tmp_path: Any) -> None:
        art = _make_artifact("entity/project/model-run123:best", "v5", "run123")
        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.resolve("entity/project", run_id="run123")

        mock_api.artifact.assert_called_once_with(
            type="model", name="entity/project/model-run123:best"
        )

    def test_uses_explicit_version_when_given(self, tmp_path: Any) -> None:
        art = _make_artifact("entity/project/model-run123:v2", "v2", "run123")
        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.resolve("entity/project", run_id="run123", version="v2")

        mock_api.artifact.assert_called_once_with(
            type="model", name="entity/project/model-run123:v2"
        )


class TestResolveViaTags:
    def test_returns_artifactref_for_best_run_by_metric(self) -> None:
        run_b = _make_run("run_b", 0.95)
        art = _make_artifact("entity/project/model-run_b:best", "v1", "run_b")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.runs.return_value = [run_b]
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            ref = registry.resolve(
                "entity/project",
                tags=["prod"],
                metric="best_val_score",
            )

        assert ref.run_id == "run_b"

    def test_passes_server_side_order_when_metric_given(self) -> None:
        run_a = _make_run("run_a", 0.9)
        art = _make_artifact("entity/project/model-run_a:best", "v1", "run_a")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.runs.return_value = [run_a]
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.resolve("entity/project", tags=["prod"], metric="best_val_score")

        call_kwargs = mock_api.runs.call_args[1]
        assert call_kwargs["order"] == "-summary_metric.best_val_score"

    def test_omits_order_kwarg_when_no_metric_given(self) -> None:
        run_a = _make_run("run_a", 0.9)
        art = _make_artifact("entity/project/model-run_a:best", "v1", "run_a")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.runs.return_value = [run_a]
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.resolve("entity/project", tags=["prod"])

        call_kwargs = mock_api.runs.call_args[1]
        assert "order" not in call_kwargs

    def test_filters_by_tags(self) -> None:
        run_a = _make_run("run_a", 0.9)
        art = _make_artifact("entity/project/model-run_a:best", "v1", "run_a")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.runs.return_value = [run_a]
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.resolve(
                "entity/project", tags=["prod", "v2"], metric="best_val_score"
            )

        call_kwargs = mock_api.runs.call_args[1]
        assert call_kwargs["filters"]["tags"] == {"$in": ["prod", "v2"]}

    def test_string_tag_is_wrapped_in_list(self) -> None:
        run_a = _make_run("run_a", 0.9)
        art = _make_artifact("entity/project/model-run_a:best", "v1", "run_a")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.runs.return_value = [run_a]
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.resolve("entity/project", tags="prod", metric="best_val_score")

        call_kwargs = mock_api.runs.call_args[1]
        assert call_kwargs["filters"]["tags"] == {"$in": ["prod"]}


class TestResolveViaRawPath:
    def test_returns_artifactref_with_run_id_from_source_run(self) -> None:
        art = _make_artifact("entity/project/model-run99:v0", "v0", "run99")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            ref = registry.resolve("entity/project/model-run99:v0")

        assert ref.run_id == "run99"
        assert ref.qualified_name == "entity/project/model-run99:v0"
        assert ref.version == "v0"
        assert ref.artifact_name == "model-run99"

    def test_returns_empty_run_id_when_logged_by_returns_none(self) -> None:
        art = MagicMock()
        art.qualified_name = "entity/project/model-orphan:v0"
        art.version = "v0"
        art.logged_by.return_value = None

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            ref = registry.resolve("entity/project/model-orphan:v0")

        assert ref.run_id == ""
        assert ref.artifact_name == "model-orphan"


class TestDownload:
    def test_returns_absolute_path_to_ckpt(self, tmp_path: Any) -> None:
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_text("fake")
        art = _make_artifact("entity/project/model-run1:v1", "v1", "run1")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art
            art.download.return_value = str(tmp_path)

            registry = WandbRegistry()
            ref = ArtifactRef(
                qualified_name="entity/project/model-run1:v1",
                run_id="run1",
                artifact_name="model-run1",
                version="v1",
            )
            result = registry.download(ref, str(tmp_path))

        assert result == str(ckpt)

    def test_raises_file_not_found_when_no_ckpt(self, tmp_path: Any) -> None:
        art = _make_artifact("entity/project/model-run1:v1", "v1", "run1")

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art
            art.download.return_value = str(tmp_path)

            registry = WandbRegistry()
            ref = ArtifactRef(
                qualified_name="entity/project/model-run1:v1",
                run_id="run1",
                artifact_name="model-run1",
                version="v1",
            )
            with pytest.raises(FileNotFoundError):
                registry.download(ref, str(tmp_path))


class TestPull:
    def test_returns_absolute_path_to_onnx(self, tmp_path: Any) -> None:
        onnx = tmp_path / "model.onnx"
        onnx.write_text("fake")
        art = MagicMock()
        art.download.return_value = str(tmp_path)

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            result = registry.pull("entity/project/model-run1:v1", str(tmp_path))

        assert result == str(onnx)

    def test_raises_file_not_found_when_no_onnx(self, tmp_path: Any) -> None:
        art = MagicMock()
        art.download.return_value = str(tmp_path)

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            with pytest.raises(FileNotFoundError):
                registry.pull("entity/project/model-run1:v1", str(tmp_path))


class TestResolveThenPullDedup:
    def test_pull_reuses_artifact_resolved_by_a_prior_resolve_call(
        self, tmp_path: Any
    ) -> None:
        onnx = tmp_path / "model.onnx"
        onnx.write_text("fake")
        art = _make_artifact("entity/project/model-run123:best", "v5", "run123")
        art.download.return_value = str(tmp_path)

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            ref = registry.resolve("entity/project", run_id="run123")
            registry.pull(artifact_path=ref.qualified_name, local_dir=str(tmp_path))

        assert mock_api.artifact.call_count == 1

    def test_pull_still_fetches_when_no_prior_resolve_matches_the_path(
        self, tmp_path: Any
    ) -> None:
        onnx = tmp_path / "model.onnx"
        onnx.write_text("fake")
        art = _make_artifact("entity/project/model-run123:best", "v5", "run123")
        art.download.return_value = str(tmp_path)

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art

            registry = WandbRegistry()
            registry.pull(
                artifact_path="entity/project/model-run123:best",
                local_dir=str(tmp_path),
            )

        assert mock_api.artifact.call_count == 1


class TestWandbNotInstalled:
    def test_resolve_raises_runtime_error_when_wandb_missing(self) -> None:
        with patch("radiologist.registry.optional._wandb", None):
            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="wandb"):
                registry.resolve("entity/project")

    def test_download_raises_runtime_error_when_wandb_missing(
        self, tmp_path: Any
    ) -> None:
        ref = ArtifactRef(
            qualified_name="entity/project/model-run1:v1",
            run_id="run1",
            artifact_name="model-run1",
            version="v1",
        )
        with patch("radiologist.registry.optional._wandb", None):
            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="wandb"):
                registry.download(ref, str(tmp_path))

    def test_pull_raises_runtime_error_when_wandb_missing(self, tmp_path: Any) -> None:
        with patch("radiologist.registry.optional._wandb", None):
            registry = WandbRegistry()
            with pytest.raises(RuntimeError, match="wandb"):
                registry.pull("entity/project/model-run1:v1", str(tmp_path))
