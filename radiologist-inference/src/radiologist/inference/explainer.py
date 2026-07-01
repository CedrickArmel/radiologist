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

"""Score-CAM explanation on top of Classifier."""

from __future__ import annotations

import json
from typing import List, Union

import numpy as np
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.base_predictor import _preprocess_image
from radiologist.inference.cam import score_cam_with_session
from radiologist.inference.classifier import Classifier
from radiologist.inference.models import Explanation


class Explainer(Classifier):
    """Adds Score-CAM explanation to Classifier."""

    def explain(self, image: Union[str, "np.ndarray", "PILImage.Image"]) -> Explanation:
        """Produce a Score-CAM saliency map for the given image.

        Args:
            image: Input as file path, HWC numpy uint8 array, or PIL Image.

        Returns:
            Explanation with a saliency map sized to the original image
            resolution and the predicted class label.
        """
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

        saliency = score_cam_with_session(
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
