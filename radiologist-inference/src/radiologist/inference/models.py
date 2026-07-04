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

"""Result and metadata dataclasses shared across the predictor hierarchy."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Inference result with class probabilities and predicted class label."""

    probabilities: Dict[str, float]
    predicted_class: str


@dataclass(frozen=True)
class Explanation:
    """Score-CAM saliency map result."""

    saliency_map: np.ndarray
    predicted_class: str


@dataclass(frozen=True)
class UncertaintyResult:
    """MC-Dropout uncertainty estimation result."""

    mean_probabilities: Dict[str, float]
    std_per_class: Dict[str, float]
    predictive_entropy: float
    n_passes: int


@dataclass(frozen=True)
class ModelMetadata:
    """ONNX model metadata extracted from session."""

    classes: List[str]
    input_shape: List[int]
    cam_target_layer: str
    output_names: List[str]
