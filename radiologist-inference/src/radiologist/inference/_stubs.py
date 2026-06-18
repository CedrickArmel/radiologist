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

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference._app import _build_app
from radiologist.inference._cam import score_cam as _score_cam
from radiologist.inference._cam import score_cam_with_session as _score_cam_with_session
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


def _read_metadata(session: ort.InferenceSession) -> Dict[str, str]:
    """Extract custom_metadata_map from an ONNX InferenceSession."""
    return dict(session.get_modelmeta().custom_metadata_map)


def _preprocess_image(
    image: Union[str, np.ndarray, PILImage.Image],
    input_shape: List[int],
) -> np.ndarray:
    """Load, resize, and normalize image to a float32 NCHW array.

    Args:
        image: File path, numpy HWC uint8 array, or PIL Image.
        input_shape: [N, C, H, W] as stored in model metadata.

    Returns:
        Float32 array of shape (1, C, H, W) with values in [0, 1].
    """
    _, _, h, w = input_shape
    if isinstance(image, str):
        pil_img = PILImage.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = PILImage.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    pil_img = pil_img.resize((w, h), PILImage.Resampling.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def _apply_prior_correction(
    softmax: np.ndarray,
    classes: List[str],
    prior: Dict[str, float],
) -> np.ndarray:
    """Scale softmax by deployment prior weights and renormalize.

    Args:
        softmax: 1-D array of class probabilities (length == len(classes)).
        classes: Ordered class names matching softmax positions.
        prior: Deployment prior probability per class name.

    Returns:
        Renormalized 1-D float32 array.
    """
    weights = np.array([prior[c] for c in classes], dtype=np.float32)
    corrected = softmax * weights
    total = corrected.sum()
    if total > 0:
        corrected = corrected / total
    return corrected


@dataclass
class _PredictorState:
    det_session: ort.InferenceSession
    metadata: Dict[str, str]
    mcd_session: Optional[ort.InferenceSession] = field(default=None)


class Predictor:
    """Facade for ONNX-backed chest X-ray classification."""

    _state: _PredictorState

    @classmethod
    def from_path(
        cls,
        det_path: str,
        mcd_path: Optional[str] = None,
    ) -> "Predictor":
        """Load Predictor from local ONNX file paths.

        Args:
            det_path: Path to the deterministic ONNX model file.
            mcd_path: Optional path to the MC-Dropout ONNX model file.

        Returns:
            Loaded Predictor instance.

        Raises:
            FileNotFoundError: If det_path does not exist.
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: If file is
                not a valid ONNX model.
        """
        try:
            det_session = ort.InferenceSession(det_path)
        except Exception:
            raise

        metadata = _read_metadata(det_session)

        mcd_session: Optional[ort.InferenceSession] = None
        if mcd_path is not None:
            mcd_session = ort.InferenceSession(mcd_path)

        instance = cls.__new__(cls)
        instance._state = _PredictorState(
            det_session=det_session,
            metadata=metadata,
            mcd_session=mcd_session,
        )
        return instance

    @classmethod
    def from_registry(
        cls,
        artifact_path: str,
        local_dir: str,
    ) -> "Predictor":
        """Download model from W&B registry and load as Predictor.

        Args:
            artifact_path: W&B artifact path (entity/project/name:version).
            local_dir: Local directory where the ONNX file will be saved.

        Returns:
            Loaded Predictor instance.

        Raises:
            RuntimeError: When the ``registry`` extra (wandb) is not installed.
        """
        det_path = pull_model(artifact_path=artifact_path, local_dir=local_dir)
        return cls.from_path(det_path=det_path)

    def predict(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        """Run deterministic inference and return class probabilities.

        Args:
            image: Input as file path, HWC numpy uint8 array, or PIL Image.
            deployment_prior: Optional per-class deployment prior probabilities.
                When supplied, overrides any embedded training prior in the model.
                When omitted, the model-embedded training_prior is used if present;
                otherwise raw softmax probabilities are returned.

        Returns:
            Prediction with per-class probabilities and predicted class label.
        """
        meta = self._state.metadata
        classes: List[str] = json.loads(meta["classes"])
        input_shape: List[int] = json.loads(meta["input_shape"])

        arr = _preprocess_image(image, input_shape)

        session = self._state.det_session
        input_name = session.get_inputs()[0].name
        outputs = session.run(["logits"], {input_name: arr})
        logits: np.ndarray = outputs[0][0]

        softmax = logits.astype(np.float64)
        softmax = softmax - softmax.max()
        softmax = np.exp(softmax)
        softmax = (softmax / softmax.sum()).astype(np.float32)

        effective_prior: Optional[Dict[str, float]] = deployment_prior
        if effective_prior is None and "training_prior" in meta:
            effective_prior = json.loads(meta["training_prior"])

        if effective_prior is not None:
            softmax = _apply_prior_correction(softmax, classes, effective_prior)

        probs = {c: float(softmax[i]) for i, c in enumerate(classes)}
        predicted = max(probs, key=probs.__getitem__)
        return Prediction(probabilities=probs, predicted_class=predicted)

    def explain(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
    ) -> Explanation:
        """Produce a Score-CAM saliency map for the given image."""
        if not hasattr(self, "_state"):
            raise NotImplementedError

        if isinstance(image, str):
            pil_orig = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_orig = PILImage.fromarray(image).convert("RGB")
        else:
            pil_orig = image.convert("RGB")
        original_w, original_h = pil_orig.size

        meta = self._state.metadata
        classes: List[str] = json.loads(meta["classes"])
        input_shape: List[int] = json.loads(meta["input_shape"])

        preprocessed = _preprocess_image(image, input_shape)

        session = self._state.det_session
        input_name = session.get_inputs()[0].name
        outputs = session.run(["logits", "feature_maps"], {input_name: preprocessed})
        logits_raw: np.ndarray = outputs[0][0]
        feature_maps: np.ndarray = outputs[1][0]

        saliency = _score_cam_with_session(
            session=session,
            preprocessed=preprocessed,
            feature_maps=feature_maps,
            original_h=original_h,
            original_w=original_w,
        )

        probs = logits_raw.astype(np.float64)
        probs = probs - probs.max()
        probs = np.exp(probs)
        probs = probs / probs.sum()
        predicted = classes[int(np.argmax(probs))]

        return Explanation(saliency_map=saliency, predicted_class=predicted)

    def predict_with_uncertainty(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        """Run MC-Dropout inference and return uncertainty estimates."""
        if self._state.mcd_session is None:
            raise RuntimeError(
                "MC-Dropout inference requires mcd_path to be supplied when"
                " loading the Predictor via from_path()."
            )
        meta = self._state.metadata
        input_shape: List[int] = json.loads(meta["input_shape"])
        arr = _preprocess_image(image, input_shape)
        return mc_dropout_predict(self._state.mcd_session, arr, n_passes=n_passes)


def pull_model(artifact_path: str, local_dir: str) -> str:
    """Download an ONNX model from the W&B Model Registry.

    Args:
        artifact_path: W&B artifact path in the form entity/project/name:version.
        local_dir: Local directory where the ONNX file will be saved.

    Returns:
        Local filesystem path to the downloaded ONNX file.

    Raises:
        RuntimeError: When the ``registry`` extra (wandb) is not installed.
        FileNotFoundError: When no .onnx file is found in the downloaded artifact.
    """
    if _wandb is None:
        raise RuntimeError(
            "The 'registry' extra is required to use pull_model. "
            "Install it with: pip install radiologist-inference[registry]"
        )
    import os

    api = _wandb.Api()
    artifact = api.artifact(artifact_path)
    download_dir = artifact.download(local_dir)

    for fname in os.listdir(download_dir):
        if fname.endswith(".onnx"):
            return os.path.join(download_dir, fname)

    raise FileNotFoundError(
        f"No .onnx file found in artifact '{artifact_path}' downloaded to"
        f" '{download_dir}'"
    )


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
    return _score_cam(feature_maps=feature_maps, logits=logits)


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
    meta = dict(session.get_modelmeta().custom_metadata_map)
    classes: List[str] = json.loads(meta["classes"])

    input_name = session.get_inputs()[0].name
    all_probs: List[np.ndarray] = []
    for _ in range(n_passes):
        outputs = session.run(["logits"], {input_name: image})
        raw: np.ndarray = outputs[0][0]
        softmax = raw.astype(np.float64)
        softmax = softmax - softmax.max()
        softmax = np.exp(softmax)
        softmax = (softmax / softmax.sum()).astype(np.float32)
        all_probs.append(softmax)

    stacked = np.stack(all_probs, axis=0)  # (n_passes, n_classes)
    mean_p = stacked.mean(axis=0)
    std_p = stacked.std(axis=0)

    epsilon = 1e-12
    entropy = float(-np.sum(mean_p * np.log(mean_p + epsilon)))

    return UncertaintyResult(
        mean_probabilities={c: float(mean_p[i]) for i, c in enumerate(classes)},
        std_per_class={c: float(std_p[i]) for i, c in enumerate(classes)},
        predictive_entropy=entropy,
        n_passes=n_passes,
    )


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
    return _build_app(_fastapi, predictor)
