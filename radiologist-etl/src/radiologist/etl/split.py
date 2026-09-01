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

"""Deterministic train/val/test split assignment via MD5 hashing."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

_MD5_MODULUS: int = 16**32

# Explicitly ordered sequence of (split name, fraction) pairs — the order is
# part of the split contract, not a formatting detail. A future issue (#184)
# rewires assign_split onto this ordered-sequence contract; today's
# dict-ratios signature below stays in place so the current, real,
# production split-assignment path keeps working.
SplitRatios = Sequence[tuple[str, float]]


def normalize_ratios(
    ratios: SplitRatios | Mapping[str, float],
) -> list[tuple[str, float]]:
    """Normalize split ratios to the explicitly ordered sequence form.

    Args:
        ratios: either the ordered ``(name, fraction)`` pair sequence, or a
            plain mapping (rejected — see Raises).

    Returns:
        The ordered pair sequence, unchanged, as a list.

    Raises:
        ValueError: if ``ratios`` is a plain mapping — split ratios must be
            an explicitly ordered sequence, because a mapping's key order is
            not a stable part of the split contract.
    """
    raise NotImplementedError


def assign_split(filename: str, ratios: dict[str, float]) -> str:
    """Deterministically assign a split label to a filename via MD5.

    Args:
        filename: image filename (basename only, not the full path).
        ratios: mapping from split name to fraction, e.g.
            ``{"train": 0.7, "val": 0.15, "test": 0.15}``.

    Returns:
        The split name this filename belongs to.

    Raises:
        ValueError: if sum(ratios.values()) deviates from 1.0 beyond float tolerance.
    """
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-6):
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios.values())!r}")
    hex_digest = hashlib.md5(filename.encode()).hexdigest()
    fraction = int(hex_digest, 16) / _MD5_MODULUS
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += ratio
        if fraction < cumulative:
            return name
    return next(reversed(ratios))
