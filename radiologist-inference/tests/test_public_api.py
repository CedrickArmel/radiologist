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

"""Tests for the radiologist.inference public API scaffold (issue #76).

All behavioral contracts are verified through the public surface only.
"""

from unittest.mock import patch

import numpy as np
import pytest


def test_package_imports_without_error():
    """Importing radiologist.inference must not raise ImportError."""
    import radiologist.inference  # noqa: F401


def test_all_public_names_present():
    """Every name listed in __all__ must be importable from the package."""
    import radiologist.inference as pkg

    expected = {
        "Predictor",
        "pull_model",
        "score_cam",
        "mc_dropout_predict",
        "Prediction",
        "Explanation",
        "UncertaintyResult",
        "ModelMetadata",
        "create_app",
    }
    assert set(pkg.__all__) == expected
    for name in expected:
        assert hasattr(pkg, name), f"Missing public name: {name}"


def test_result_dataclasses_are_importable():
    """Result dataclasses must be importable and constructable with correct fields."""
    from radiologist.inference import (
        Explanation,
        ModelMetadata,
        Prediction,
        UncertaintyResult,
    )

    pred = Prediction(probabilities={"a": 1.0}, predicted_class="a")
    assert pred.predicted_class == "a"

    exp = Explanation(
        saliency_map=np.zeros((4, 4), dtype=np.float32), predicted_class="a"
    )
    assert exp.predicted_class == "a"

    unc = UncertaintyResult(
        mean_probabilities={"a": 1.0},
        std_per_class={"a": 0.0},
        predictive_entropy=0.0,
        n_passes=30,
    )
    assert unc.n_passes == 30

    meta = ModelMetadata(
        classes=["a", "b"],
        input_shape=[1, 3, 224, 224],
        cam_target_layer="features",
        output_names=["output"],
        mc_dropout=False,
    )
    assert meta.classes == ["a", "b"]


def test_predictor_from_registry_raises_not_implemented():
    """Predictor.from_registry must raise NotImplementedError (not yet implemented)."""
    from radiologist.inference import Predictor

    with pytest.raises(NotImplementedError):
        Predictor.from_registry(
            artifact_path="entity/project/name:v0", local_dir="/tmp"
        )


def test_predictor_explain_raises_not_implemented():
    """Predictor.explain must raise NotImplementedError (not yet implemented)."""
    from radiologist.inference import Predictor

    predictor = object.__new__(Predictor)
    with pytest.raises(NotImplementedError):
        predictor.explain(image=np.zeros((224, 224, 3), dtype=np.uint8))


def test_predictor_predict_with_uncertainty_raises_not_implemented():
    """Predictor.predict_with_uncertainty raises NotImplementedError."""
    from radiologist.inference import Predictor

    predictor = object.__new__(Predictor)
    with pytest.raises(NotImplementedError):
        predictor.predict_with_uncertainty(
            image=np.zeros((224, 224, 3), dtype=np.uint8)
        )


def test_score_cam_returns_saliency_map_in_0_1():
    """score_cam must return a numpy array with all values in [0, 1]."""
    from radiologist.inference import score_cam

    result = score_cam(
        feature_maps=np.random.rand(64, 7, 7).astype(np.float32),
        logits=np.array([0.3, 0.7], dtype=np.float32),
    )
    assert isinstance(result, np.ndarray)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_mc_dropout_predict_raises_not_implemented():
    """mc_dropout_predict must raise NotImplementedError (not yet implemented)."""
    from radiologist.inference import mc_dropout_predict

    with pytest.raises(NotImplementedError):
        mc_dropout_predict(
            session=object(),
            image=np.zeros((1, 3, 224, 224), dtype=np.float32),
        )


def test_pull_model_raises_runtime_error_when_wandb_absent():
    """pull_model raises RuntimeError naming 'registry' when wandb is absent."""
    import radiologist.inference._stubs as stubs

    with patch.object(stubs, "_wandb", None):
        with pytest.raises(RuntimeError, match="registry"):
            stubs.pull_model(artifact_path="entity/project/name:v0", local_dir="/tmp")


def test_create_app_raises_runtime_error_when_fastapi_absent():
    """create_app raises RuntimeError naming 'serve' when fastapi is absent."""
    import radiologist.inference._stubs as stubs

    with patch.object(stubs, "_fastapi", None):
        with pytest.raises(RuntimeError, match="serve"):
            stubs.create_app()
