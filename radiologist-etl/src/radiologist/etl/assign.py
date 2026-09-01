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

"""Assign-split stage: extract manifests folder → deduped, split-assigned manifest.

Reads every extract manifest in a folder, concatenates and deduplicates by
source path (warning on duplicates), assigns each record's split
deterministically from its filename alone, and writes a single split
manifest. This stage never uses a runner — it always runs locally.
"""

from __future__ import annotations

from radiologist.etl.models import AssignSplitResult
from radiologist.etl.split import SplitRatios


def assign_splits(
    manifests_dir: str,
    destination: str,
    ratios: SplitRatios | None = None,
    run_label: str | None = None,
    storage_options: dict | None = None,
) -> AssignSplitResult:
    """Concatenate, dedupe, and split-assign every extract manifest in a folder.

    Args:
        manifests_dir: folder of extract manifests, read in sorted name order.
        destination: folder the split manifest is written into.
        ratios: explicitly ordered split ratios; defaults to
            ``[("train", 0.70), ("val", 0.15), ("test", 0.15)]``.
        run_label: optional label folded into the run id.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        An :class:`~radiologist.etl.models.AssignSplitResult` describing the run.

    Raises:
        FileNotFoundError: if the folder holds no manifest.
    """
    raise NotImplementedError
