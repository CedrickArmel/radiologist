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

"""Score-CAM implementation (issue #79)."""

from typing import List

import numpy as np
from PIL import Image as PILImage  # type: ignore[import-untyped]


def score_cam(
    feature_maps: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    """Compute a Score-CAM saliency map from feature maps via GAP weighting.

    Uses global-average-pooling of each channel as the channel weight —
    the approximation path used when no ONNX session is available for
    masked forward passes. ``logits`` is accepted for interface symmetry
    with ``score_cam_with_session`` but does not affect the result.

    Args:
        feature_maps: Feature maps of shape (C, H, W).
        logits: Unused GAP-approximation fallback argument, kept for
            interface symmetry with score_cam_with_session.

    Returns:
        Saliency map of shape (H, W) with values in [0, 1].
    """
    _, h, w = feature_maps.shape
    weights = feature_maps.mean(axis=(1, 2))
    cam = np.tensordot(weights, feature_maps, axes=([0], [0]))

    cam = np.maximum(cam, 0.0)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros((h, w), dtype=np.float32)
    return ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)


def score_cam_with_session(
    session: object,
    preprocessed: np.ndarray,
    feature_maps: np.ndarray,
    original_h: int,
    original_w: int,
) -> np.ndarray:
    """Run full Score-CAM with masked forward passes against an ONNX session.

    Args:
        session: onnxruntime.InferenceSession for the deterministic model.
        preprocessed: Float32 NCHW array of shape (1, C_in, H_in, W_in).
        feature_maps: Feature maps of shape (C, H_f, W_f).
        original_h: Height of original image for output resizing.
        original_w: Width of original image for output resizing.

    Returns:
        Saliency map of shape (original_h, original_w) with values in [0, 1].
    """
    import onnxruntime as ort  # type: ignore[import-untyped]

    sess: ort.InferenceSession = session  # type: ignore[assignment]
    input_name = sess.get_inputs()[0].name
    _, _, h_in, w_in = preprocessed.shape
    n_channels, feat_h, feat_w = feature_maps.shape

    logits_full = sess.run(["logits"], {input_name: preprocessed})[0][0]
    predicted_class = int(np.argmax(logits_full))

    scores: List[float] = []
    for i in range(n_channels):
        channel = feature_maps[i]
        ch_min = float(channel.min())
        ch_max = float(channel.max())
        if ch_max - ch_min < 1e-8:
            normalized = np.zeros((h_in, w_in), dtype=np.float32)
        else:
            norm_ch = (channel - ch_min) / (ch_max - ch_min)
            pil_ch = PILImage.fromarray((norm_ch * 255).astype(np.uint8))
            pil_up = pil_ch.resize((w_in, h_in), PILImage.Resampling.BILINEAR)
            normalized = np.array(pil_up, dtype=np.float32) / 255.0

        mask = normalized[np.newaxis, np.newaxis, :, :]
        masked_logits = sess.run(["logits"], {input_name: preprocessed * mask})[0][0]
        scores.append(float(masked_logits[predicted_class]))

    scores_arr = np.array(scores, dtype=np.float32)
    cam = np.tensordot(scores_arr, feature_maps, axes=([0], [0]))

    cam = np.maximum(cam, 0.0)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros((original_h, original_w), dtype=np.float32)

    cam_norm = ((cam - cam_min) / (cam_max - cam_min) * 255).astype(np.uint8)
    pil_cam = PILImage.fromarray(cam_norm)
    pil_cam_resized = pil_cam.resize(
        (original_w, original_h), PILImage.Resampling.BILINEAR
    )
    return np.array(pil_cam_resized, dtype=np.float32) / 255.0
