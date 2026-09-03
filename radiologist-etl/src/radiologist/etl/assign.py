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

This stage lists its input folder twice per run: once (via
:func:`~radiologist.etl.identity.compute_assign_run_id`) to fingerprint it
for the run id, and once here to enumerate the manifests it then reads. That
is a deliberate, accepted tradeoff — two calls, not a per-file stat storm —
not an oversight; collapsing them would couple two otherwise single-purpose
modules for a saving that is not measurable at this stage's cadence.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import fsspec  # type: ignore[import-untyped]

from radiologist.etl.identity import (
    EXTRACT_MANIFEST_PREFIX,
    EXTRACT_MANIFEST_SUFFIX,
    _matches,
    compute_assign_run_id,
)
from radiologist.etl.manifest import JsonlWriter, ManifestRecord, records_reader
from radiologist.etl.models import AssignSplitResult
from radiologist.etl.split import SplitRatios, assign_split, normalize_ratios

logger = logging.getLogger(__name__)

# Shipped default order/values — every filename keeps the split the previous
# (dict-ratios) pipeline gave it. Changing this order or these fractions
# re-partitions every corpus and must be treated as a breaking data change.
_DEFAULT_RATIOS: SplitRatios = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def assign_splits(
    manifests_dir: str,
    destination: str,
    ratios: SplitRatios | None = None,
    run_label: str | None = None,
    storage_options: dict | None = None,
) -> AssignSplitResult:
    """Concatenate, dedupe, and split-assign every extract manifest in a folder.

    Args:
        manifests_dir: folder scanned for ``extract-``-prefixed ``.jsonl``
            manifests, read in sorted name order. Anything else in the folder —
            including a split manifest this stage wrote on a previous run — is
            ignored, so ``destination`` may safely equal ``manifests_dir``.
        destination: folder the split manifest is written into.
        ratios: explicitly ordered split ratios; defaults to
            ``[("train", 0.70), ("val", 0.15), ("test", 0.15)]``.
        run_label: optional label folded into the run id.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        An :class:`~radiologist.etl.models.AssignSplitResult` describing the run.

    Raises:
        FileNotFoundError: if the folder holds no manifest.
        ValueError: see :func:`~radiologist.etl.split.normalize_ratios`.
    """
    ordered_ratios = normalize_ratios(ratios if ratios is not None else _DEFAULT_RATIOS)

    fs, path = fsspec.url_to_fs(manifests_dir, **(storage_options or {}))
    try:
        entries = fs.ls(path, detail=True)
    except FileNotFoundError:
        entries = []
    # Selects exactly the set compute_assign_run_id fingerprints — the same
    # shared predicate, applied to the same two constants. Sorting the full
    # names (not the basenames) keeps multi-manifest merge order unchanged
    # for existing corpora.
    manifest_files = sorted(
        str(entry["name"])
        for entry in entries
        if _matches(entry, EXTRACT_MANIFEST_PREFIX, EXTRACT_MANIFEST_SUFFIX)
    )
    if not manifest_files:
        raise FileNotFoundError(
            f"No {EXTRACT_MANIFEST_PREFIX!r} manifest found in {manifests_dir!r}"
        )

    seen_paths: dict[str, ManifestRecord] = {}
    seen_filenames: dict[str, str] = {}
    duplicate_count = 0
    records: list[ManifestRecord] = []
    collided_filenames: set[str] = set()
    for name in manifest_files:
        uri = fs.unstrip_protocol(name)
        for record in records_reader(uri, storage_options=storage_options):
            if record.path in seen_paths:
                duplicate_count += 1
                continue
            seen_paths[record.path] = record
            records.append(record)
            prior_path = seen_filenames.get(record.filename)
            if prior_path is None:
                seen_filenames[record.filename] = record.path
            elif (
                prior_path != record.path and record.filename not in collided_filenames
            ):
                collided_filenames.add(record.filename)
                logger.warning(
                    "Filename collision across distinct source paths: %r "
                    "(a shared filename would collide as a shard key downstream)",
                    record.filename,
                )
    if duplicate_count:
        logger.warning(
            "Dropped %d duplicate record(s) sharing a source path", duplicate_count
        )

    config = {
        "ratios": [list(pair) for pair in ordered_ratios],
        "run_label": run_label,
    }
    run_id = compute_assign_run_id(
        manifests_dir, config, storage_options=storage_options
    )

    counts_by_split: dict[str, int] = {name: 0 for name, _ in ordered_ratios}
    final_records: list[ManifestRecord] = []
    for record in records:
        split = assign_split(record.filename, ordered_ratios)
        counts_by_split[split] += 1
        final_records.append(replace(record, split=split, manifest_id=run_id))

    split_manifest_path = f"{destination.rstrip('/')}/manifest-{run_id}.jsonl"
    JsonlWriter().write(
        final_records, split_manifest_path, storage_options=storage_options
    )

    return AssignSplitResult(
        run_id=run_id,
        split_manifest_path=split_manifest_path,
        source_manifest_count=len(manifest_files),
        record_count=len(final_records),
        duplicate_count=duplicate_count,
        counts_by_split=counts_by_split,
    )
