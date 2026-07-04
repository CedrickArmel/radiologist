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

"""Manifest record dataclass and Parquet/JSONL reader-writer helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

_NON_STAT_COLS = frozenset(
    {
        "manifest_id",
        "path",
        "filename",
        "label",
        "split",
        "shard",
        "lung_out_of_frame",
        "excluded",
        "exclusion_reason",
    }
)


@dataclass
class ManifestRecord:
    """A single row in the ETL manifest.

    Args:
        manifest_id: 16-char run ID, same for all records in a run.
        path: full URI to the source image.
        filename: image file name.
        label: class label.
        split: dataset split (train/val/test).
        stats: Haralick/asymmetry feature dict; flattened at write time.
        lung_out_of_frame: True when the lung is cropped out; None when masks unavailable.
        excluded: whether the record should be excluded from training.
        exclusion_reason: pipe-joined reason codes for exclusion.
        shard: WebDataset shard name; None until assigned by shards.py.
    """

    manifest_id: str
    path: str
    filename: str
    label: str
    split: str
    stats: dict[str, float]
    lung_out_of_frame: bool | None = None
    excluded: bool = False
    exclusion_reason: str = ""
    shard: str | None = None

    def _to_flat_dict(self) -> dict:
        """Return a flat dict with stats spread into top-level keys."""
        d: dict = {
            "manifest_id": self.manifest_id,
            "path": self.path,
            "filename": self.filename,
            "label": self.label,
            "split": self.split,
            "shard": self.shard,
            **self.stats,
            "lung_out_of_frame": self.lung_out_of_frame,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> ManifestRecord:
        """Reconstruct a ManifestRecord from a flat dict (JSON or pandas NamedTuple row).

        Args:
            d: flat dict with all ManifestRecord fields; stat columns are any keys
               not in _NON_STAT_COLS.

        Returns:
            A ManifestRecord instance.
        """
        stats = {k: float(v) for k, v in d.items() if k not in _NON_STAT_COLS}

        def _is_na(v: object) -> bool:
            if v is None:
                return True
            try:
                import pandas as pd

                return bool(pd.isna(v))
            except (TypeError, ValueError):
                return False

        loof_raw = d.get("lung_out_of_frame")
        loof: bool | None = None if _is_na(loof_raw) else bool(loof_raw)

        shard_raw = d.get("shard")
        shard: str | None = None if _is_na(shard_raw) else str(shard_raw)

        return cls(
            manifest_id=str(d["manifest_id"]),
            path=str(d["path"]),
            filename=str(d["filename"]),
            label=str(d["label"]),
            split=str(d.get("split", "") or ""),
            stats=stats,
            lung_out_of_frame=loof,
            excluded=bool(d.get("excluded", False)),
            exclusion_reason=str(d.get("exclusion_reason", "") or ""),
            shard=shard,
        )


class ParquetWriter:
    """Write :class:`ManifestRecord` lists to Parquet via fsspec."""

    def write(
        self,
        records: list[ManifestRecord],
        destination: str,
        storage_options: dict | None = None,
    ) -> None:
        """Write records to a Parquet file via fsspec.

        Args:
            records: list of ManifestRecord instances.
            destination: local path or remote URI.
            storage_options: extra kwargs for fsspec.
        """
        if not records:
            raise ValueError("Cannot write an empty manifest — no records to persist.")

        flat_rows = [r._to_flat_dict() for r in records]

        # Collect stat columns from the first record (all records share the same keys)
        stat_columns = list(records[0].stats.keys())

        # Build schema to ensure nullable columns are typed correctly
        base_fields = [
            pa.field("manifest_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("label", pa.string()),
            pa.field("split", pa.string()),
            pa.field("shard", pa.string(), nullable=True),
        ]
        stat_fields = [pa.field(col, pa.float64()) for col in stat_columns]
        tail_fields = [
            pa.field("lung_out_of_frame", pa.bool_(), nullable=True),
            pa.field("excluded", pa.bool_()),
            pa.field("exclusion_reason", pa.string()),
        ]
        schema = pa.schema(base_fields + stat_fields + tail_fields)

        table = pa.Table.from_pylist(flat_rows, schema=schema)

        fs, path = fsspec.url_to_fs(destination, **(storage_options or {}))
        if hasattr(fs, "makedirs"):
            fs.makedirs(str(Path(path).parent), exist_ok=True)
        with fs.open(destination, "wb") as f:
            pq.write_table(table, f)


class JsonlWriter:
    """Write :class:`ManifestRecord` lists to JSONL via fsspec."""

    def write(
        self,
        records: list[ManifestRecord],
        destination: str,
        storage_options: dict | None = None,
    ) -> None:
        """Write records to a JSONL file via fsspec (one JSON object per line).

        Args:
            records: list of ManifestRecord instances.
            destination: local path or remote URI.
            storage_options: extra kwargs for fsspec.
        """
        fs, path = fsspec.url_to_fs(destination, **(storage_options or {}))
        if hasattr(fs, "makedirs"):
            fs.makedirs(str(Path(path).parent), exist_ok=True)
        with fs.open(destination, "wt", encoding="utf-8") as f:
            for record in records:
                flat = record._to_flat_dict()
                f.write(json.dumps(flat) + "\n")


def records_reader(path: str, storage_options) -> list[ManifestRecord]:
    """Read a JSONL manifest back into a list of :class:`ManifestRecord`.

    Args:
        path: local path or remote URI to the JSONL manifest.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        List of ManifestRecord instances, one per non-empty line, in file order.
    """
    fs, mpath = fsspec.url_to_fs(path, **storage_options)
    records: list[ManifestRecord] = []
    with fs.open(mpath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ManifestRecord.from_flat_dict(json.loads(line)))
    return records
