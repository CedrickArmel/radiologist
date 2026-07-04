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

"""Deterministic classification on top of BasePredictor."""

from __future__ import annotations

import json
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.base_predictor import (
    BasePredictor,
    _apply_prior_correction,
    _preprocess_image,
    _softmax,
)
from radiologist.inference.models import Prediction


class Classifier(BasePredictor):
    """Adds deterministic classification to BasePredictor."""

    def predict(
        self,
        image: Union[str, "np.ndarray", "PILImage.Image"],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        """Run deterministic inference and return class probabilities.

        Args:
            image: Input as file path, HWC numpy uint8 array, or PIL Image.
            deployment_prior: Optional per-class deployment prior probabilities.
                When supplied, overrides any embedded training prior in the
                model. When omitted, the model-embedded training_prior is
                used if present; otherwise raw softmax probabilities are
                returned.

        Returns:
            Prediction with per-class probabilities and predicted class label.
        """
        model_metadata = self._state.model_metadata
        input_shape = model_metadata.input_shape

        arr = _preprocess_image(
            image, input_shape, mean=self._state.mean, std=self._state.std
        )

        session = self._state.session
        input_name = session.get_inputs()[0].name
        outputs = session.run(["logits"], {input_name: arr})
        logits: np.ndarray = outputs[0][0]

        return self._predict_from_logits(logits, deployment_prior)

    def _predict_from_logits(
        self,
        logits: "np.ndarray",
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        """Turn raw model logits into a prior-corrected Prediction.

        Shared by predict() and Explainer.explain() so predicted_class always
        agrees between the two, regardless of which one triggered inference.
        """
        meta = self._state.metadata
        classes = self._state.model_metadata.classes

        softmax = _softmax(logits)

        effective_prior: Optional[Dict[str, float]] = deployment_prior
        if effective_prior is None and "training_prior" in meta:
            effective_prior = json.loads(meta["training_prior"])

        if effective_prior is not None:
            softmax = _apply_prior_correction(softmax, classes, effective_prior)

        probs = {c: float(softmax[i]) for i, c in enumerate(classes)}
        predicted = max(probs, key=probs.__getitem__)
        return Prediction(probabilities=probs, predicted_class=predicted)
