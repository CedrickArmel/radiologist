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

import json
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.base_predictor import (
    BasePredictor,
    _preprocess_image,
    _read_metadata,
    _resolve_and_pull,
    _softmax,
)
from radiologist.inference.models import UncertaintyResult
from radiologist.registry.wandb_registry import WandbRegistry

if TYPE_CHECKING:
    from radiologist.registry.interface import ModelRegistry
    from radiologist.registry.selector import RegistrySelector


class MCDropoutPredictor(BasePredictor):
    """Adds MC-Dropout uncertainty estimation to BasePredictor."""

    @classmethod
    def from_selector(
        cls,
        selector: "RegistrySelector",
        local_dir: str,
        registry: Optional["ModelRegistry"] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        input_shape: Optional[List[int]] = None,
    ) -> "MCDropoutPredictor":
        """Resolve det + MC-Dropout ({run_id}-mcd) artifacts and load via from_path.

        Args:
            selector: Selector for the deterministic artifact. When
                selector.run_id is set, the MC-Dropout counterpart is looked
                up at f"{run_id}-mcd"; otherwise the same selector
                (tags/groups/metric) is reused for both, matching today's CLI
                fallback behavior.
            local_dir: Local directory where both ONNX files will be saved.
            registry: Registry to resolve/download from. Defaults to a single
                shared WandbRegistry() instance when omitted.
            mean: Optional normalization mean, forwarded to from_path.
            std: Optional normalization std, forwarded to from_path.
            input_shape: Optional input_shape fallback, forwarded to
                from_path.

        Returns:
            Loaded MCDropoutPredictor instance.

        Raises:
            RuntimeError: When the ``registry`` extra (wandb) is not
                installed.
        """
        reg = registry if registry is not None else WandbRegistry()
        det_path = _resolve_and_pull(selector, local_dir, reg)
        mcd_run_id = f"{selector.run_id}-mcd" if selector.run_id else selector.run_id
        mcd_selector = replace(selector, run_id=mcd_run_id)
        mcd_path = _resolve_and_pull(mcd_selector, local_dir, reg)
        return cls.from_path(
            det_path=det_path,
            mcd_path=mcd_path,
            mean=mean,
            std=std,
            input_shape=input_shape,
        )

    def predict_with_uncertainty(
        self,
        image: Union[str, "np.ndarray", "PILImage.Image"],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        """Run MC-Dropout inference and return uncertainty estimates.

        Raises:
            RuntimeError: When no stochastic (MC-Dropout) session was loaded.
        """
        mcd_session = self._state.mcd_session
        if mcd_session is None:
            raise RuntimeError(
                "MC-Dropout inference requires mcd_path to be supplied when"
                " loading the predictor via from_path()."
            )
        input_shape = self._state.model_metadata.input_shape
        arr = _preprocess_image(
            image, input_shape, mean=self._state.mean, std=self._state.std
        )
        return mc_dropout_predict(mcd_session, arr, n_passes=n_passes)


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
    meta = _read_metadata(session)
    classes: List[str] = json.loads(meta["classes"])

    input_name = session.get_inputs()[0].name
    all_probs: List["np.ndarray"] = []
    for _ in range(n_passes):
        outputs = session.run(["logits"], {input_name: image})
        raw: "np.ndarray" = outputs[0][0]
        all_probs.append(_softmax(raw))

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
