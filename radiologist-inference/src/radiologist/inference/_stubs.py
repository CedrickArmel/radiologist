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

"""Public API stubs for radiologist.inference (issue #76).

All entry points raise NotImplementedError until implemented in later issues.
Entry points that require optional extras raise RuntimeError naming the extra
when that extra is absent.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference._optional import _fastapi, _typer, _wandb  # noqa: F401


@dataclass
class Prediction:
    """Inference result with class probabilities and predicted class label."""

    probabilities: Dict[str, float]
    predicted_class: str


@dataclass
class Explanation:
    """Score-CAM saliency map result."""

    saliency_map: np.ndarray
    predicted_class: str


@dataclass
class UncertaintyResult:
    """MC-Dropout uncertainty estimation result."""

    mean_probabilities: Dict[str, float]
    std_per_class: Dict[str, float]
    predictive_entropy: float
    n_passes: int


@dataclass
class ModelMetadata:
    """ONNX model metadata extracted from session."""

    classes: List[str]
    input_shape: List[int]
    cam_target_layer: str
    output_names: List[str]
    mc_dropout: bool


class Predictor:
    """Facade for ONNX-backed chest X-ray classification."""

    @classmethod
    def from_path(
        cls,
        det_path: str,
        mcd_path: Optional[str] = None,
    ) -> "Predictor":
        """Load Predictor from local ONNX file paths."""
        raise NotImplementedError

    @classmethod
    def from_registry(
        cls,
        artifact_path: str,
        local_dir: str,
    ) -> "Predictor":
        """Download model from W&B registry and load as Predictor."""
        raise NotImplementedError

    def predict(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        """Run deterministic inference and return class probabilities."""
        raise NotImplementedError

    def explain(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
    ) -> Explanation:
        """Produce a Score-CAM saliency map for the given image."""
        raise NotImplementedError

    def predict_with_uncertainty(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        """Run MC-Dropout inference and return uncertainty estimates."""
        raise NotImplementedError


def pull_model(artifact_path: str, local_dir: str) -> str:
    """Download an ONNX model from the W&B Model Registry.

    Args:
        artifact_path: W&B artifact path in the form entity/project/name:version.
        local_dir: Local directory where the ONNX file will be saved.

    Returns:
        Local filesystem path to the downloaded ONNX file.

    Raises:
        RuntimeError: When the ``registry`` extra (wandb) is not installed.
    """
    if _wandb is None:
        raise RuntimeError(
            "The 'registry' extra is required to use pull_model. "
            "Install it with: pip install radiologist-inference[registry]"
        )
    raise NotImplementedError


def score_cam(
    feature_maps: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    """Compute a Score-CAM saliency map from feature maps and logits.

    Args:
        feature_maps: Feature maps of shape (C, H, W).
        logits: Model logits of shape (num_classes,).

    Returns:
        Saliency map of shape (H, W) with values in [0, 1].
    """
    raise NotImplementedError


def mc_dropout_predict(
    session: Any,
    image: np.ndarray,
    n_passes: int = 30,
) -> UncertaintyResult:
    """Run stochastic MC-Dropout forward passes and aggregate uncertainty.

    Args:
        session: ONNX Runtime InferenceSession for the MC-Dropout model.
        image: Preprocessed input array matching the model's input shape.
        n_passes: Number of stochastic forward passes.

    Returns:
        UncertaintyResult with mean probabilities, per-class std, entropy, and
        pass count.
    """
    raise NotImplementedError


def create_app(predictor: Optional["Predictor"] = None) -> Any:
    """Create and return the FastAPI application instance.

    Args:
        predictor: Optional Predictor instance to inject at startup.

    Returns:
        FastAPI application instance.

    Raises:
        RuntimeError: When the ``serve`` extra (fastapi, uvicorn) is not installed.
    """
    if _fastapi is None:
        raise RuntimeError(
            "The 'serve' extra is required to use create_app. "
            "Install it with: pip install radiologist-inference[serve]"
        )
    raise NotImplementedError
