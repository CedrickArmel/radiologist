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

"""MC-Dropout uncertainty estimation on top of BasePredictor."""

from __future__ import annotations

from typing import Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.base_predictor import BasePredictor
from radiologist.inference.models import UncertaintyResult


class MCDropoutPredictor(BasePredictor):
    """Adds MC-Dropout uncertainty estimation to BasePredictor."""

    def predict_with_uncertainty(
        self,
        image: Union[str, "np.ndarray", "PILImage.Image"],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        """Run MC-Dropout inference and return uncertainty estimates.

        Raises:
            RuntimeError: When no stochastic (MC-Dropout) session was loaded.
        """
        raise NotImplementedError


def mc_dropout_predict(
    session: "ort.InferenceSession", image: "np.ndarray", n_passes: int = 30
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
