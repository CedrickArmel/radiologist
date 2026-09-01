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

"""Parallel stats computation over an image root directory."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import numpy as np
from rich.progress import Progress

from radiologist.etl.manifest import ManifestRecord
from radiologist.etl.models import BatchOutcome
from radiologist.etl.stats import StatExtractor
from radiologist.utils import Logger, read_image

SUPPORTED_FORMATS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})

logger = Logger(__name__)


def _resolve_mask(
    image_path: str,
    images_root: str,
    masks_root: str | None,
    storage_options: dict | None,
) -> np.ndarray | None:
    """Resolve the mask array for an image by mirroring its path under masks_root.

    Args:
        image_path: full URI or path to the source image.
        images_root: root directory of all images; used to compute relative path.
        masks_root: root directory of masks; None when masks are unavailable.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Mask array, or None if masks_root is None or the mask file is missing.
    """
    if masks_root is None:
        return None

    # Normalize both paths to bare filesystem paths (strips file:// if present)
    _, norm_image = fsspec.url_to_fs(image_path, **(storage_options or {}))
    _, norm_root = fsspec.url_to_fs(images_root, **(storage_options or {}))

    rel_raw = norm_image[len(norm_root) :]
    if rel_raw and not rel_raw.startswith("/"):
        # norm_root is a string prefix of norm_image but not a parent directory
        return None
    rel = rel_raw.lstrip("/")
    mask_path = masks_root.rstrip("/") + "/" + rel
    try:
        arr, _ = read_image(mask_path, storage_options=storage_options)
        return arr
    except FileNotFoundError:
        return None


def lung_out_of_frame(mask: np.ndarray) -> bool:
    """Check if any nonzero mask pixel touches the image border.

    Args:
        mask: 2-D or 3-D (H, W) or (H, W, C) boolean/integer array.

    Returns:
        True if any nonzero pixel is on the first/last row or column.
    """
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return bool(border.any())


def _process_one(
    image_path: str,
    images_root: str,
    masks_root: str | None,
    manifest_id: str,
    extractors: list[StatExtractor],
    storage_options: dict | None,
) -> ManifestRecord:
    """Process a single image into a ManifestRecord.

    Top-level function required for ProcessPoolExecutor picklability.

    Args:
        image_path: full URI to the source image.
        images_root: root directory used to resolve the mask mirror path.
        masks_root: root directory of masks; None when masks are unavailable.
        manifest_id: run identifier shared by all records in a run.
        extractors: list of StatExtractor callables.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A ManifestRecord populated with stats and lung_out_of_frame.
    """
    arr, meta = read_image(image_path, storage_options=storage_options)
    mask = _resolve_mask(image_path, images_root, masks_root, storage_options)

    stats: dict[str, float] = {}
    for extractor in extractors:
        stats.update(extractor(arr, meta, mask=mask))

    loof = lung_out_of_frame(mask) if mask is not None else None
    label = Path(image_path).parent.name
    filename = Path(image_path).name

    return ManifestRecord(
        manifest_id=manifest_id,
        path=image_path,
        filename=filename,
        label=label,
        split="",
        stats=stats,
        lung_out_of_frame=loof,
    )


def process_batch(
    paths: Sequence[str],
    images_root: str | None,
    masks_root: str | None,
    manifest_id: str,
    extractors: list[StatExtractor],
    storage_options: dict | None = None,
) -> BatchOutcome:
    """Process one batch of image paths into a :class:`BatchOutcome`.

    Top-level (picklable) function required for process-pool dispatch. This
    is the extract stage's per-batch worker; it replaces
    :class:`StatsProcessor` for the new explicit-file-listing extract flow
    (issue #183 implements the body). ``StatsProcessor`` itself stays in
    place, unchanged, because the current monolithic ``etl_flow`` still
    depends on it.

    Args:
        paths: image paths to process in this batch.
        images_root: root directory used to resolve mask mirror paths.
        masks_root: root directory of masks; None when masks are unavailable.
        manifest_id: run identifier stamped on every produced record.
        extractors: list of StatExtractor callables.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A :class:`~radiologist.etl.models.BatchOutcome`: one record per
        readable image, one ``(path, message)`` entry per unreadable one.
        Never raises for a single bad image.
    """
    raise NotImplementedError


class StatsProcessor:
    """Run stat extraction over an image collection using a process pool.

    Args:
        extractors: list of StatExtractor callables to apply to each image.
        workers: number of worker processes; defaults to 1.
    """

    def __init__(
        self,
        extractors: list[StatExtractor],
        workers: int | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            extractors: list of StatExtractor callables to apply to each image.
            workers: number of worker processes; defaults to 1 when None.
        """
        self._extractors = extractors
        self._workers = workers or 1

    def run(
        self,
        source: str,
        manifest_id: str,
        masks_root: str | None = None,
        storage_options: dict | None = None,
    ) -> list[ManifestRecord]:
        """Process all images reachable from source.

        Args:
            source: fsspec-compatible URI to the image root directory.
            manifest_id: run identifier stamped on every ManifestRecord.
            masks_root: optional root directory for segmentation masks.
            storage_options: extra kwargs forwarded to fsspec.

        Returns:
            List of ManifestRecord, one per successfully processed image.
            Failed images are logged and skipped.
        """
        logger.info("Processing images to extract statistics...")

        fs, root = fsspec.url_to_fs(source, **(storage_options or {}))
        all_paths = sorted(fs.find(root))

        image_paths = [
            fs.unstrip_protocol(p)
            for p in all_paths
            if Path(p).suffix.lower() in SUPPORTED_FORMATS
        ]

        records: list[ManifestRecord] = []

        with ProcessPoolExecutor(
            max_workers=self._workers, mp_context=mp.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    p,
                    source,
                    masks_root,
                    manifest_id,
                    self._extractors,
                    storage_options,
                ): p
                for p in image_paths
            }
            with Progress() as progress:
                task_id = progress.add_task("Processing images...", total=len(futures))
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        records.append(future.result())
                    except Exception as exc:
                        logger.warning("Skipping %r: %s", path, exc)
                    finally:
                        progress.advance(task_id)
        logger.info("Statistics extraction completed successfully!")
        return records
