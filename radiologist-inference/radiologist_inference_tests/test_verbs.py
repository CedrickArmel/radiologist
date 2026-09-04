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

"""Tests for the predictor-verb registry (verbs.py).

Registry-backed cases mock the wandb/registry process boundary only
(patch radiologist.registry.resolver._wandb); from_path/from_selector run
for real against ONNX models built by build_det_onnx/build_mcd_onnx.
"""

from unittest.mock import MagicMock, patch

import pytest
from _helpers import build_det_onnx, build_mcd_onnx

from radiologist.inference import Classifier, Explainer, MCDropoutPredictor
from radiologist.inference.verbs import apply_mcd_convention, get_verb, load_predictor


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


class TestGetVerb:
    def test_uncertainty_verb_constructs_mcdropout_predictor_with_mcd_convention(self):
        verb = get_verb("uncertainty")
        assert verb.predictor_cls is MCDropoutPredictor
        assert verb.mcd_convention is True

    def test_predict_verb_constructs_classifier_without_mcd_convention(self):
        verb = get_verb("predict")
        assert verb.predictor_cls is Classifier
        assert verb.mcd_convention is False

    def test_explain_verb_constructs_explainer_without_mcd_convention(self):
        verb = get_verb("explain")
        assert verb.predictor_cls is Explainer
        assert verb.mcd_convention is False

    def test_unregistered_verb_name_raises_key_error(self):
        with pytest.raises(KeyError):
            get_verb("nonexistent-verb")


class TestApplyMcdConvention:
    def test_truthy_run_id_maps_to_run_id_dash_mcd_suffix(self):
        assert apply_mcd_convention("run1") == "run1-mcd"

    def test_none_run_id_maps_to_none(self):
        assert apply_mcd_convention(None) is None

    def test_empty_string_run_id_maps_to_none(self):
        assert apply_mcd_convention("") is None


class TestLoadPredictorFromLocalPath:
    def test_predict_verb_with_local_model_path_returns_classifier(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        verb = get_verb("predict")

        predictor = load_predictor(
            verb,
            path=det_path,
            run_id=None,
            tags=None,
            groups=None,
            metric=None,
            local_dir=str(tmp_path),
        )

        assert isinstance(predictor, Classifier)

    def test_explain_verb_with_local_model_path_returns_explainer(self, tmp_path):
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        verb = get_verb("explain")

        predictor = load_predictor(
            verb,
            path=det_path,
            run_id=None,
            tags=None,
            groups=None,
            metric=None,
            local_dir=str(tmp_path),
        )

        assert isinstance(predictor, Explainer)

    def test_uncertainty_verb_with_local_model_path_returns_mcdropout_predictor(
        self, tmp_path
    ):
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        verb = get_verb("uncertainty")

        predictor = load_predictor(
            verb,
            path=mcd_path,
            run_id=None,
            tags=None,
            groups=None,
            metric=None,
            local_dir=str(tmp_path),
        )

        assert isinstance(predictor, MCDropoutPredictor)


class TestLoadPredictorFromRegistry:
    def test_predict_verb_with_path_and_run_id_resolves_classifier_from_registry(
        self, tmp_path
    ):
        build_det_onnx(tmp_path, filename="det.onnx")
        verb = get_verb("predict")
        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            predictor = load_predictor(
                verb,
                path="entity/project",
                run_id="run1",
                tags=None,
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        assert isinstance(predictor, Classifier)
        api_instance = mock_wandb.Api.return_value
        resolve_calls = [
            call for call in api_instance.artifact.call_args_list if call.kwargs
        ]
        assert len(resolve_calls) == 1
        assert resolve_calls[0].kwargs["name"].startswith("entity/project/")

    def test_uncertainty_verb_with_path_and_run_id_resolves_against_mcd_suffixed_run_id(
        self, tmp_path
    ):
        build_mcd_onnx(tmp_path, filename="mcd.onnx")
        verb = get_verb("uncertainty")
        mock_wandb, pulled_art = _make_registry_wandb_mock(
            qualified_name="entity/project/model-run1-mcd:best"
        )
        pulled_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            predictor = load_predictor(
                verb,
                path="entity/project",
                run_id="run1",
                tags=None,
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        assert isinstance(predictor, MCDropoutPredictor)
        api_instance = mock_wandb.Api.return_value
        resolve_calls = [
            call for call in api_instance.artifact.call_args_list if call.kwargs
        ]
        assert len(resolve_calls) == 1
        assert "run1-mcd" in resolve_calls[0].kwargs["name"]
        assert resolve_calls[0].kwargs["name"].startswith("entity/project/")


class TestLoadPredictorRaisesWhenNoSourceGiven:
    def test_no_path_and_no_registry_selector_raises_value_error(self, tmp_path):
        verb = get_verb("predict")

        with pytest.raises(ValueError) as exc_info:
            load_predictor(
                verb,
                path=None,
                run_id=None,
                tags=None,
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        message = str(exc_info.value)
        assert "--path" in message
        assert "--run-id" in message
        assert "--tags" in message
        assert "--groups" in message
        assert "--metric" in message


class TestLoadPredictorUsesPathAsRegistryOverride:
    def test_path_and_run_id_together_resolves_from_registry_under_path_entity_project(
        self, tmp_path
    ):
        build_det_onnx(tmp_path, filename="det.onnx")
        verb = get_verb("predict")
        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            predictor = load_predictor(
                verb,
                path="entity/project",
                run_id="run1",
                tags=None,
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        assert isinstance(predictor, Classifier)
        api_instance = mock_wandb.Api.return_value
        resolve_calls = [
            call for call in api_instance.artifact.call_args_list if call.kwargs
        ]
        assert len(resolve_calls) == 1
        assert resolve_calls[0].kwargs["name"].startswith("entity/project/")

    def test_path_and_tags_together_resolves_from_registry_under_path_entity_project(
        self, tmp_path
    ):
        build_det_onnx(tmp_path, filename="det.onnx")
        verb = get_verb("predict")
        mock_wandb, pulled_art = _make_registry_wandb_mock()
        pulled_art.download.return_value = str(tmp_path)

        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            predictor = load_predictor(
                verb,
                path="entity/project",
                run_id=None,
                tags=["a"],
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        assert isinstance(predictor, Classifier)
        api_instance = mock_wandb.Api.return_value
        resolve_calls = [
            call for call in api_instance.artifact.call_args_list if call.kwargs
        ]
        assert len(resolve_calls) == 1
        assert resolve_calls[0].kwargs["name"].startswith("entity/project/")


class TestLoadPredictorRequiresPathWithSelector:
    def test_run_id_without_path_raises_value_error_naming_path(self, tmp_path):
        verb = get_verb("predict")

        with pytest.raises(ValueError) as exc_info:
            load_predictor(
                verb,
                path=None,
                run_id="run1",
                tags=None,
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        message = str(exc_info.value)
        assert "--path" in message

    def test_tags_without_path_raises_value_error_naming_path(self, tmp_path):
        verb = get_verb("predict")

        with pytest.raises(ValueError) as exc_info:
            load_predictor(
                verb,
                path=None,
                run_id=None,
                tags=["a"],
                groups=None,
                metric=None,
                local_dir=str(tmp_path),
            )

        message = str(exc_info.value)
        assert "--path" in message

    def test_groups_without_path_raises_value_error_naming_path(self, tmp_path):
        verb = get_verb("predict")

        with pytest.raises(ValueError) as exc_info:
            load_predictor(
                verb,
                path=None,
                run_id=None,
                tags=None,
                groups=["g"],
                metric=None,
                local_dir=str(tmp_path),
            )

        message = str(exc_info.value)
        assert "--path" in message

    def test_metric_without_path_raises_value_error_naming_path(self, tmp_path):
        verb = get_verb("predict")

        with pytest.raises(ValueError) as exc_info:
            load_predictor(
                verb,
                path=None,
                run_id=None,
                tags=None,
                groups=None,
                metric="f1",
                local_dir=str(tmp_path),
            )

        message = str(exc_info.value)
        assert "--path" in message
