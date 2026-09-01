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

"""Deterministic train/val/test split assignment via MD5 hashing.

Split ratios are an explicitly ordered sequence of ``(name, fraction)``
pairs — the bracket order is a stated part of the split contract, not an
accident of how a YAML mapping happened to be written. A plain mapping is
rejected outright: coercing it (e.g. by sorting its keys) would restore
exactly the hidden order-dependence this contract removes, and would
silently re-partition every already-processed corpus once for no gain.

**Guaranteed property**: a filename's split is a pure function of the
filename and this ordered ratio sequence alone — never of the corpus size,
the filesystem's listing order, or any stored state. Changing the shipped
default order or fractions re-partitions every corpus and must be treated
as a breaking data change.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

_MD5_MODULUS: int = 16**32

# Explicitly ordered sequence of (split name, fraction) pairs — the order is
# part of the split contract, not a formatting detail.
SplitRatios = Sequence[tuple[str, float]]


def normalize_ratios(
    ratios: SplitRatios | Mapping[str, float],
) -> list[tuple[str, float]]:
    """Validate and normalize split ratios to the ordered sequence form.

    Args:
        ratios: either the ordered ``(name, fraction)`` pair sequence, or a
            plain mapping (rejected — see Raises).

    Returns:
        The ordered pair sequence, unchanged, as a list.

    Raises:
        ValueError: if ``ratios`` is a plain mapping (order is part of the
            split contract and a mapping's key order is not a stable part
            of it); if the sequence is empty; if it contains a repeated
            split name; if any fraction is negative; or if the fractions do
            not sum to ``1.0`` (within float tolerance).
    """
    if isinstance(ratios, Mapping):
        raise ValueError(
            "Split ratios must be an explicitly ordered sequence of "
            "(name, fraction) pairs, not a mapping — ratio order is part "
            "of the split contract, and a mapping's key order is not a "
            "stable part of it."
        )
    pairs = list(ratios)
    if not pairs:
        raise ValueError("Split ratios must not be an empty sequence.")
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"Split ratios contain a repeated split name: {names!r}")
    for name, fraction in pairs:
        if fraction < 0:
            raise ValueError(
                f"Split ratios must not contain a negative fraction: "
                f"{name!r}={fraction!r}"
            )
    total = sum(fraction for _, fraction in pairs)
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total!r}")
    return pairs


def assign_split(filename: str, ratios: SplitRatios) -> str:
    """Deterministically assign a split label to a filename via MD5.

    A pure function of ``filename`` and ``ratios`` alone — no stored state
    is consulted, so a filename's assignment cannot depend on which other
    filenames are present in the corpus.

    Args:
        filename: image filename (basename only, not the full path).
        ratios: explicitly ordered ``(name, fraction)`` pairs, e.g.
            ``[("train", 0.70), ("val", 0.15), ("test", 0.15)]``.

    Returns:
        The split name this filename belongs to.

    Raises:
        ValueError: see :func:`normalize_ratios`.
    """
    pairs = normalize_ratios(ratios)
    hex_digest = hashlib.md5(filename.encode()).hexdigest()
    fraction = int(hex_digest, 16) / _MD5_MODULUS
    cumulative = 0.0
    for name, ratio in pairs:
        cumulative += ratio
        if fraction < cumulative:
            return name
    return pairs[-1][0]
