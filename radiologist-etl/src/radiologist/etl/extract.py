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

"""Extract stage: explicit file listing → stats + quality filters → batch manifest.

Processes exactly the images named in an explicit file listing (no directory
rescanning), extracts Haralick GLCM + lung asymmetry stats, applies IQR and
lung-out-of-frame quality filters, and writes one batch manifest per run.
The split column is left empty — split assignment is the assign-split
stage's responsibility.
"""

from __future__ import annotations

from radiologist.etl.execution import BatchMapper
from radiologist.etl.models import ExtractResult
from radiologist.etl.stats import StatExtractor


class ExtractionFailureError(RuntimeError):
    """Raised when the share of unreadable images exceeds max_failure_rate."""


def read_file_list(
    file_list: str,
    storage_options: dict | None = None,
) -> list[str]:
    """Read a newline-delimited listing of image URIs.

    Args:
        file_list: fsspec-compatible URI to the listing file.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        URIs in file order; blank lines and lines starting with ``#`` are
        skipped; surrounding whitespace is stripped.

    Raises:
        FileNotFoundError: if the listing itself is absent.
        ValueError: if the listing yields no entries.
    """
    raise NotImplementedError


def extract(
    file_list: str,
    destination: str,
    images_root: str | None = None,
    masks_root: str | None = None,
    extractors: list[StatExtractor] | None = None,
    iqr_columns: list[str] | None = None,
    iqr_factor: float = 1.5,
    workers: int | None = None,
    batch_size: int = 64,
    max_failure_rate: float = 0.0,
    run_label: str | None = None,
    mapper: BatchMapper | None = None,
    storage_options: dict | None = None,
) -> ExtractResult:
    """Run the extract stage over exactly the images named in ``file_list``.

    Args:
        file_list: fsspec-compatible URI to the newline-delimited listing.
        destination: folder that accumulates extract manifests.
        images_root: root directory of source images; required when
            ``masks_root`` is set.
        masks_root: optional root directory for segmentation masks.
        extractors: feature extractors; defaults to Haralick + lung asymmetry.
        iqr_columns: column names to test for IQR outliers.
        iqr_factor: IQR multiplier for the fence.
        workers: worker count; defaults to :func:`~radiologist.etl.execution.default_workers`.
        batch_size: images per dispatched batch.
        max_failure_rate: maximum tolerated ``failed / total`` before raising.
        run_label: optional label folded into the run id.
        mapper: batch-dispatching callable; defaults to a local process-pool mapper.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        An :class:`~radiologist.etl.models.ExtractResult` describing the run.

    Raises:
        ValueError: if ``masks_root`` is set without ``images_root``.
        ExtractionFailureError: if ``failed / total`` exceeds ``max_failure_rate``.
    """
    raise NotImplementedError
