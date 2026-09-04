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

"""Base class for ONNX-backed predictors: loading, preprocessing, and priors.

Concrete inference verbs (predict / explain / predict_with_uncertainty) live
in the subclasses defined in classifier.py, explainer.py, and mc_dropout.py.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.models import ModelMetadata
from radiologist.registry.models import ArtifactRef
from radiologist.registry.selector import resolve_selector
from radiologist.registry.wandb_registry import WandbRegistry

if TYPE_CHECKING:
    from radiologist.registry.interface import ModelRegistry
    from radiologist.registry.selector import RegistrySelector

_INFERENCE_WANDB_MISSING_MSG = (
    "wandb is required for registry operations. "
    "Install with: pip install 'radiologist-inference[registry]'"
)


@dataclass
class _PredictorState:
    session: ort.InferenceSession
    metadata: Dict[str, str]
    model_metadata: ModelMetadata
    mean: Optional[float] = field(default=None)
    std: Optional[float] = field(default=None)
    provenance: Optional[ArtifactRef] = field(default=None)
    # contract: set only by from_selector; None for from_path-loaded predictors


class BasePredictor:
    """Common loading and preprocessing surface for all predictor classes."""

    _state: _PredictorState

    @property
    def provenance(self) -> Optional[ArtifactRef]:
        """The ArtifactRef this predictor was resolved from.

        Returns:
            The resolved ``ArtifactRef``, or ``None`` when this predictor was
            loaded from a local file path rather than a registry selector.
        """
        return self._state.provenance

    @classmethod
    def from_path(
        cls,
        model_path: str,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        input_shape: Optional[List[int]] = None,
    ) -> "BasePredictor":
        """Load a predictor instance from a local ONNX file path.

        Args:
            model_path: Path to the ONNX model file. For an MCDropoutPredictor
                this must be the stochastic (MC-Dropout) model — the single
                session state is verb-agnostic and holds whichever model was
                loaded.
            mean: Optional normalization mean. When omitted (with std also
                omitted), preprocessing keeps today's /255.0-only scaling.
            std: Optional normalization std. Must be provided together with
                mean.
            input_shape: Optional [N, C, H, W] fallback used when the ONNX
                file's metadata has no input_shape key.

        Returns:
            Loaded instance of the calling subclass.

        Raises:
            FileNotFoundError: If model_path does not exist.
            ValueError: If exactly one of mean/std is provided, or if no
                input_shape can be resolved from metadata or the argument.
        """
        _validate_mean_std(mean, std)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No such ONNX model file: {model_path}")
        session = ort.InferenceSession(model_path)
        metadata = _read_metadata(session)

        model_metadata = _parse_model_metadata(
            metadata,
            default_input_shape=input_shape,
        )

        instance = cls.__new__(cls)
        instance._state = _PredictorState(
            session=session,
            metadata=metadata,
            model_metadata=model_metadata,
            mean=mean,
            std=std,
        )
        return instance

    @classmethod
    def from_registry(
        cls,
        artifact_path: str,
        local_dir: str,
        registry: Optional["ModelRegistry"] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        input_shape: Optional[List[int]] = None,
    ) -> "BasePredictor":
        """Download a model from a registry and load it via from_path.

        Args:
            artifact_path: Registry artifact path (entity/project/name:version).
            local_dir: Local directory where the ONNX file will be saved.
            registry: Registry to pull from. Defaults to WandbRegistry() when
                omitted.
            mean: Optional normalization mean, forwarded to from_path.
            std: Optional normalization std, forwarded to from_path.
            input_shape: Optional input_shape fallback, forwarded to
                from_path.

        Returns:
            Loaded instance of the calling subclass.

        Raises:
            RuntimeError: When the ``registry`` extra (wandb) is not installed.
            ValueError: If exactly one of mean/std is provided.
        """
        _validate_mean_std(mean, std)
        reg = registry if registry is not None else WandbRegistry()
        with _translate_wandb_missing():
            model_path = reg.pull(artifact_path=artifact_path, local_dir=local_dir)
        return cls.from_path(
            model_path=model_path, mean=mean, std=std, input_shape=input_shape
        )

    @classmethod
    def from_selector(
        cls,
        selector: "RegistrySelector",
        local_dir: str,
        registry: Optional["ModelRegistry"] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        input_shape: Optional[List[int]] = None,
    ) -> "BasePredictor":
        """Resolve a selector against a registry and load via from_path.

        Args:
            selector: Declarative description of which artifact to resolve.
            local_dir: Local directory where the ONNX file will be saved.
            registry: Registry to resolve/download from. Defaults to
                WandbRegistry() when omitted.
            mean: Optional normalization mean, forwarded to from_path.
            std: Optional normalization std, forwarded to from_path.
            input_shape: Optional input_shape fallback, forwarded to
                from_path.

        Returns:
            Loaded instance of the calling subclass.

        Raises:
            RuntimeError: When the ``registry`` extra (wandb) is not installed.
            ValueError: If exactly one of mean/std is provided.
        """
        _validate_mean_std(mean, std)
        model_path, resolved_ref = _resolve_and_pull(selector, local_dir, registry)
        instance = cls.from_path(
            model_path=model_path, mean=mean, std=std, input_shape=input_shape
        )
        instance._state.provenance = resolved_ref
        return instance


def _resolve_and_pull(
    selector: "RegistrySelector",
    local_dir: str,
    registry: Optional["ModelRegistry"] = None,
) -> Tuple[str, ArtifactRef]:
    """Resolve a selector to an artifact ref, then pull its ONNX file."""
    reg = registry if registry is not None else WandbRegistry()
    with _translate_wandb_missing():
        ref = resolve_selector(selector, reg)
        local_path = reg.pull(artifact_path=ref.qualified_name, local_dir=local_dir)
    return local_path, ref


@contextmanager
def _translate_wandb_missing() -> Generator[None, None, None]:
    """Translate a missing-wandb RuntimeError into the inference-extra message."""
    try:
        yield
    except RuntimeError as exc:
        raise RuntimeError(_INFERENCE_WANDB_MISSING_MSG) from exc


def _validate_mean_std(mean: Optional[float], std: Optional[float]) -> None:
    """Ensure mean and std are either both given or both omitted.

    Args:
        mean: Optional normalization mean.
        std: Optional normalization std.

    Raises:
        ValueError: If exactly one of mean/std is provided.
    """
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be provided together")


def _read_metadata(session: "ort.InferenceSession") -> Dict[str, str]:
    """Extract custom_metadata_map from an ONNX InferenceSession."""
    return dict(session.get_modelmeta().custom_metadata_map)


def _parse_model_metadata(
    metadata: Dict[str, str],
    default_input_shape: Optional[List[int]] = None,
) -> ModelMetadata:
    """Parse the raw ONNX metadata dict into a typed ModelMetadata.

    Args:
        metadata: Raw custom_metadata_map from the loaded session.
        default_input_shape: Fallback [N, C, H, W] used when the metadata has
            no input_shape key.

    Returns:
        Typed ModelMetadata with JSON fields decoded.

    Raises:
        ValueError: If the metadata has no input_shape key and
            default_input_shape is not provided.
    """
    if "input_shape" in metadata:
        input_shape = json.loads(metadata["input_shape"])
    else:
        input_shape = default_input_shape
    if input_shape is None:
        raise ValueError(
            "ONNX model has no input_shape metadata; pass input_shape" " explicitly."
        )
    return ModelMetadata(
        classes=json.loads(metadata["classes"]),
        input_shape=input_shape,
        cam_target_layer=metadata.get("cam_target_layer", "null"),
        output_names=json.loads(metadata.get("output_names", "[null]")),
    )


def _to_pil(image: Union[str, "np.ndarray", "PILImage.Image"]) -> "PILImage.Image":
    """Convert a file path, numpy HWC uint8 array, or PIL Image to RGB PIL.

    Args:
        image: File path, numpy HWC uint8 array, or PIL Image.

    Returns:
        RGB-converted PIL Image.
    """
    if isinstance(image, str):
        return PILImage.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        return PILImage.fromarray(image).convert("RGB")
    return image.convert("RGB")


def _normalize_pil(
    pil_img: "PILImage.Image",
    input_shape: List[int],
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> "np.ndarray":
    """Resize and normalize an already-decoded PIL image to a float32 NCHW array.

    Args:
        pil_img: RGB-converted PIL Image.
        input_shape: [N, C, H, W] as stored in model metadata.
        mean: Optional normalization mean. When omitted (with std also
            omitted), the array is left in [0, 1] (today's default).
        std: Optional normalization std. Must be provided together with mean.

    Returns:
        Float32 array of shape (1, C, H, W): raw [0, 1] scale by default, or
        (arr - mean) / std when both mean and std are given.

    Raises:
        ValueError: If exactly one of mean/std is provided.
    """
    _validate_mean_std(mean, std)
    _, _, h, w = input_shape
    pil_img = pil_img.resize((w, h), PILImage.Resampling.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    if mean is not None and std is not None:
        arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr


def _preprocess_image(
    image: Union[str, "np.ndarray", "PILImage.Image"],
    input_shape: List[int],
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> "np.ndarray":
    """Load, resize, and normalize image to a float32 NCHW array.

    Args:
        image: File path, numpy HWC uint8 array, or PIL Image.
        input_shape: [N, C, H, W] as stored in model metadata.
        mean: Optional normalization mean, forwarded to _normalize_pil.
        std: Optional normalization std, forwarded to _normalize_pil.

    Returns:
        Float32 array of shape (1, C, H, W).
    """
    return _normalize_pil(_to_pil(image), input_shape, mean=mean, std=std)


def _softmax(logits: "np.ndarray") -> "np.ndarray":
    """Numerically stable softmax over the last axis.

    Args:
        logits: Raw model output logits.

    Returns:
        Float32 array of the same shape with values summing to 1.
    """
    shifted = logits.astype(np.float64) - logits.astype(np.float64).max()
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)


def _apply_prior_correction(
    softmax: "np.ndarray", classes: List[str], prior: Dict[str, float]
) -> "np.ndarray":
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
