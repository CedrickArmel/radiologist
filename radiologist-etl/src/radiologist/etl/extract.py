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

import functools
from typing import Any

import fsspec  # type: ignore[import-untyped]
import pandas as pd

from radiologist.etl.execution import (
    BatchMapper,
    chunked,
    default_workers,
    local_mapper,
)
from radiologist.etl.filters import filter_iqr, filter_lung_out_of_frame
from radiologist.etl.identity import compute_extract_run_id
from radiologist.etl.manifest import JsonlWriter, ManifestRecord
from radiologist.etl.models import ExtractResult
from radiologist.etl.processors import (
    _MASKS_ROOT_REQUIRES_IMAGES_ROOT,
    process_batch,
)
from radiologist.etl.stats import StatExtractor, lung_asymmetry, make_haralick


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
    fs, path = fsspec.url_to_fs(file_list, **(storage_options or {}))
    if not fs.exists(path):
        raise FileNotFoundError(f"No such file or directory: {file_list!r}")

    with fs.open(path, "rt", encoding="utf-8") as f:
        lines = f.readlines()

    entries = [
        stripped
        for stripped in (line.strip() for line in lines)
        if stripped and not stripped.startswith("#")
    ]
    if not entries:
        raise ValueError(f"file listing {file_list!r} resolved to no entries")
    return entries


def _extractor_signature(extractor: StatExtractor) -> dict[str, Any]:
    """Build a stable, JSON-serializable identity for a StatExtractor.

    ``functools.partial``'s own ``repr``/``str`` embeds the wrapped
    function's memory address, which would make the run id unstable across
    processes. This introspects ``.func``/``.args``/``.keywords`` when
    present (as produced by :func:`~radiologist.etl.stats.make_haralick`)
    and falls back to the callable's qualified name.
    """
    func = getattr(extractor, "func", extractor)
    args = getattr(extractor, "args", ()) or ()
    keywords = getattr(extractor, "keywords", {}) or {}
    return {
        "name": f"{func.__module__}.{func.__qualname__}",
        "args": list(args),
        "kwargs": dict(keywords),
    }


def _default_extractors() -> list[StatExtractor]:
    return [make_haralick(), lung_asymmetry]


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
    if masks_root is not None and images_root is None:
        raise ValueError(_MASKS_ROOT_REQUIRES_IMAGES_ROOT)

    paths = read_file_list(file_list, storage_options=storage_options)

    resolved_extractors = (
        list(extractors) if extractors is not None else _default_extractors()
    )
    resolved_iqr_columns = (
        list(iqr_columns) if iqr_columns is not None else ["haralick_mean"]
    )
    resolved_workers = workers if workers is not None else default_workers()

    config: dict[str, Any] = {
        "images_root": images_root,
        "masks_root": masks_root,
        "extractors": [_extractor_signature(e) for e in resolved_extractors],
        "iqr_columns": resolved_iqr_columns,
        "iqr_factor": iqr_factor,
        "run_label": run_label,
    }
    run_id = compute_extract_run_id(file_list, config, storage_options=storage_options)
    manifest_path = f"{destination.rstrip('/')}/extract-{run_id}.jsonl"

    batches = chunked(paths, batch_size)

    resolved_mapper: BatchMapper = (
        mapper
        if mapper is not None
        else local_mapper(
            functools.partial(
                process_batch,
                images_root=images_root,
                masks_root=masks_root,
                manifest_id=run_id,
                extractors=resolved_extractors,
                storage_options=storage_options,
            ),
            workers=resolved_workers,
        )
    )

    outcomes = resolved_mapper(batches)

    records: list[ManifestRecord] = []
    failures: list[tuple[str, str]] = []
    for outcome in outcomes:
        records.extend(outcome.records)
        failures.extend(outcome.failures)

    # Guarantee every record carries this run's id regardless of which
    # mapper produced it — an injected mapper (e.g. a flow's mapped-task
    # wrapper) cannot know run_id until this function has computed it, so
    # it may stamp a placeholder. The default local mapper already binds
    # the correct value, making this a no-op re-assignment for that path.
    for record in records:
        record.manifest_id = run_id

    total = len(paths)
    failed = len(failures)
    succeeded = len(records)
    failure_rate = failed / total if total else 0.0

    if failure_rate > max_failure_rate:
        failure_desc = "; ".join(f"{p!r} ({msg})" for p, msg in failures)
        raise ExtractionFailureError(
            f"extract stage failed: {failed}/{total} image(s) unreadable "
            f"(failure rate {failure_rate:.2%} exceeds max_failure_rate "
            f"{max_failure_rate:.2%}): {failure_desc}"
        )

    _apply_quality_filters(records, resolved_iqr_columns, iqr_factor)
    excluded = sum(1 for r in records if r.excluded)

    JsonlWriter().write(records, manifest_path, storage_options=storage_options)

    return ExtractResult(
        run_id=run_id,
        manifest_path=manifest_path,
        total=total,
        succeeded=succeeded,
        failed=failed,
        failure_rate=failure_rate,
        excluded=excluded,
    )


def _apply_quality_filters(
    records: list[ManifestRecord],
    iqr_columns: list[str],
    iqr_factor: float,
) -> None:
    """Run IQR + lung-out-of-frame filters over the assembled batch of records.

    Mutates each record's ``excluded``/``exclusion_reason`` in place. The
    IQR fence is a property of the whole listed batch, so this only runs
    after every batch has completed.
    """
    if not records:
        return

    rows = []
    for record in records:
        row: dict[str, Any] = dict(record.stats)
        row["lung_out_of_frame"] = record.lung_out_of_frame
        row["excluded"] = record.excluded
        row["exclusion_reason"] = record.exclusion_reason
        rows.append(row)

    df = pd.DataFrame(rows)
    present_columns = [col for col in iqr_columns if col in df.columns]
    if present_columns:
        df = filter_iqr(df, present_columns, iqr_factor)
    df = filter_lung_out_of_frame(df)

    for record, (_, row) in zip(records, df.iterrows()):
        record.excluded = bool(row["excluded"])
        record.exclusion_reason = str(row["exclusion_reason"])
