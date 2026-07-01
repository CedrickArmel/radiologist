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

from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image as PILImage  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from radiologist.registry.interface import ModelRegistry


class BasePredictor:
    """Common loading and preprocessing surface for all predictor classes."""

    @classmethod
    def from_path(
        cls, det_path: str, mcd_path: Optional[str] = None
    ) -> "BasePredictor":
        """Load a predictor instance from local ONNX file paths.

        Args:
            det_path: Path to the deterministic ONNX model file.
            mcd_path: Optional path to the MC-Dropout ONNX model file.

        Returns:
            Loaded instance of the calling subclass.

        Raises:
            FileNotFoundError: If det_path does not exist.
        """
        raise NotImplementedError

    @classmethod
    def from_registry(
        cls,
        artifact_path: str,
        local_dir: str,
        registry: Optional["ModelRegistry"] = None,
    ) -> "BasePredictor":
        """Download a model from a registry and load it via from_path.

        Args:
            artifact_path: Registry artifact path (entity/project/name:version).
            local_dir: Local directory where the ONNX file will be saved.
            registry: Registry to pull from. Defaults to WandbRegistry() when
                omitted.

        Returns:
            Loaded instance of the calling subclass.

        Raises:
            RuntimeError: When the ``registry`` extra (wandb) is not installed.
        """
        raise NotImplementedError


def _read_metadata(session: "ort.InferenceSession") -> Dict[str, str]:
    """Extract custom_metadata_map from an ONNX InferenceSession."""
    raise NotImplementedError


def _preprocess_image(
    image: Union[str, "np.ndarray", "PILImage.Image"],
    input_shape: List[int],
) -> "np.ndarray":
    """Load, resize, and normalize image to a float32 NCHW array.

    Args:
        image: File path, numpy HWC uint8 array, or PIL Image.
        input_shape: [N, C, H, W] as stored in model metadata.

    Returns:
        Float32 array of shape (1, C, H, W) with values in [0, 1].
    """
    raise NotImplementedError


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
    raise NotImplementedError
