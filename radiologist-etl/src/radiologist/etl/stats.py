# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
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

from __future__ import annotations

import functools
import math
from typing import Protocol

import numpy as np

HARALICK_PROPERTIES: frozenset[str] = frozenset(
    {
        "mean",
        "std",
        "entropy",
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    }
)

_DEFAULT_DISTANCES: list[int] = [1]
_DEFAULT_ANGLES: list[float] = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]


class StatExtractor(Protocol):
    def __call__(
        self,
        image: np.ndarray,
        metadata: dict[str, str],
        mask: np.ndarray | None = None,
    ) -> dict[str, float]: ...


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Normalize float or convert RGB image to uint8 grayscale.

    Args:
        image: H x W or H x W x C array of any numeric dtype.

    Returns:
        H x W uint8 array.
    """
    if image.ndim == 3:
        # Weighted luminance: ITU-R BT.601
        gray = (
            0.299 * image[:, :, 0].astype(np.float64)
            + 0.587 * image[:, :, 1].astype(np.float64)
            + 0.114 * image[:, :, 2].astype(np.float64)
        )
    else:
        gray = image.astype(np.float64)

    if gray.dtype != np.uint8:
        lo, hi = gray.min(), gray.max()
        if hi > lo:
            gray = (gray - lo) / (hi - lo) * 255.0
        else:
            gray = np.zeros_like(gray)
    return gray.astype(np.uint8)


def _haralick_extractor(
    image: np.ndarray,
    metadata: dict[str, str],
    mask: np.ndarray | None = None,
    features: list[str] | None = None,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> dict[str, float]:
    """Compute Haralick GLCM features for a single image.

    scikit-image is imported here; it must not be imported at module level.

    Args:
        image: H x W or H x W x C array.
        metadata: arbitrary string key-value pairs (unused internally).
        mask: optional mask (unused; present for protocol compliance).
        features: feature names to compute.
        distances: pixel-pair distances for GLCM.
        angles: angles in radians.

    Returns:
        Dict mapping ``haralick_{feature}`` to mean scalar over all
        (distance, angle) pairs.
    """
    from skimage.feature import (  # type: ignore[import-untyped]
        graycomatrix,
        graycoprops,
    )

    resolved_features: list[str] = (
        list(features) if features is not None else list(HARALICK_PROPERTIES)
    )
    resolved_distances: list[int] = (
        distances if distances is not None else _DEFAULT_DISTANCES
    )
    resolved_angles: list[float] = angles if angles is not None else _DEFAULT_ANGLES

    gray = _to_uint8_gray(image)
    glcm = graycomatrix(
        gray,
        distances=resolved_distances,
        angles=resolved_angles,
        symmetric=True,
        normed=True,
    )

    result: dict[str, float] = {}
    for feat in resolved_features:
        props = graycoprops(glcm, feat)
        result[f"haralick_{feat}"] = float(props.mean())
    return result


def make_haralick(
    features: list[str] | None = None,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
) -> StatExtractor:
    """Build a Haralick GLCM feature extractor.

    Args:
        features: subset of HARALICK_PROPERTIES to compute; defaults to all nine.
        distances: pixel-pair distances for GLCM; defaults to [1].
        angles: angles in radians; defaults to [0, π/4, π/2, 3π/4].

    Returns:
        A picklable callable matching the StatExtractor protocol.

    Raises:
        ValueError: if any name in features is not in HARALICK_PROPERTIES.
    """
    resolved: list[str] = (
        list(features) if features is not None else list(HARALICK_PROPERTIES)
    )
    unknown = [f for f in resolved if f not in HARALICK_PROPERTIES]
    if unknown:
        raise ValueError(f"unknown Haralick feature(s): {unknown!r}")
    return functools.partial(
        _haralick_extractor,
        features=resolved,
        distances=distances,
        angles=angles,
    )


def lung_asymmetry(
    image: np.ndarray,
    metadata: dict[str, str],
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute left/right lung asymmetry metrics from a segmentation mask.

    Args:
        image: H x W or H x W x C image array (unused beyond signature compliance).
        metadata: arbitrary string key-value pairs (unused).
        mask: binary or integer mask; if None returns empty dict.

    Returns:
        Dict with keys ``asymmetry_ratio`` and ``asymmetry_diff``, or ``{}`` if
        mask is None.
    """
    if mask is None:
        return {}

    if mask.ndim == 3:
        mask2d = mask.max(axis=-1)
    else:
        mask2d = mask

    midcol = mask2d.shape[1] // 2
    left = int(np.count_nonzero(mask2d[:, :midcol]))
    right = int(np.count_nonzero(mask2d[:, midcol:]))

    total = left + right
    lo = min(left, right)
    hi = max(left, right)

    ratio = float(hi) / float(lo) if lo > 0 else 1.0
    diff = float(hi - lo) / float(total) if total > 0 else 0.0

    return {"asymmetry_ratio": ratio, "asymmetry_diff": diff}
