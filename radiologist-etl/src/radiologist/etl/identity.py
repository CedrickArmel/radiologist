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

"""Content and config digests, and per-stage content-addressed run-id computation.

Each stage computes its own 16-character run id by hashing its input's
content together with only the config that affects its output — execution
backend settings (workers, batch size, runner selection) are never part of
the digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import fsspec  # type: ignore[import-untyped]

# Filename prefix and suffix the extract stage writes
# ("{destination}/extract-{run_id}.jsonl"). Single source of truth for "this
# file is an extract manifest", shared by the assign-split folder scan and the
# assign-split run-id fingerprint so the two can never disagree.
EXTRACT_MANIFEST_PREFIX: str = "extract-"
EXTRACT_MANIFEST_SUFFIX: str = ".jsonl"


def _not_found(uri: str) -> FileNotFoundError:
    return FileNotFoundError(f"No such file or directory: {uri!r}")


def _matches(entry: Mapping[str, Any], prefix: str, suffix: str) -> bool:
    """True when a listing entry is a file whose basename matches prefix/suffix.

    The one predicate ``directory_digest`` and the assign-split folder scan
    both use, so the rule for "what counts as a matching manifest" can never
    diverge between the run-id fingerprint and the stage that reads the files.
    """
    if entry.get("type") != "file":
        return False
    basename = str(entry["name"]).rsplit("/", 1)[-1]
    return basename.startswith(prefix) and basename.endswith(suffix)


def content_digest(
    uri: str,
    storage_options: dict | None = None,
    chunk_size: int = 1048576,
) -> str:
    """Stream a SHA-256 digest of a single object's bytes.

    Args:
        uri: fsspec-compatible URI to the object.
        storage_options: extra kwargs forwarded to fsspec.
        chunk_size: number of bytes read per chunk while streaming.

    Returns:
        The full 64-char hex digest.

    Raises:
        FileNotFoundError: if the object does not exist.
    """
    fs, path = fsspec.url_to_fs(uri, **(storage_options or {}))
    digest = hashlib.sha256()
    try:
        with fs.open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise _not_found(uri) from exc
    return digest.hexdigest()


def config_digest(config: Mapping[str, Any]) -> str:
    """Hash a config mapping via canonical (sort_keys) JSON.

    Args:
        config: mapping of output-affecting configuration values.

    Returns:
        The full 64-char hex digest; key order never affects the result.
    """
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directory_digest(
    directory: str,
    suffix: str = ".jsonl",
    storage_options: dict | None = None,
    prefix: str = "",
) -> str:
    """Hash the sorted (basename, size) pairs of a directory's matching entries.

    Only the final path component of each entry is hashed, so a byte-identical
    folder copied to another location yields the same digest.

    Args:
        directory: fsspec-compatible URI to the directory.
        suffix: only entries whose basename ends with this suffix are included.
        storage_options: extra kwargs forwarded to fsspec.
        prefix: only entries whose basename starts with this prefix are
            included; ``""`` matches on suffix alone.

    Returns:
        The full 64-char hex digest, obtained with a single detailed listing
        call — never one stat call per entry.
    """
    fs, path = fsspec.url_to_fs(directory, **(storage_options or {}))
    try:
        entries = fs.ls(path, detail=True)
    except FileNotFoundError as exc:
        raise _not_found(directory) from exc

    pairs = sorted(
        (str(entry["name"]).rsplit("/", 1)[-1], entry.get("size"))
        for entry in entries
        if _matches(entry, prefix, suffix)
    )
    payload = json.dumps(pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_run_id(stage: str, input_digest: str, cfg_digest: str) -> str:
    """Combine a stage name with its input and config digests into a 16-char id.

    Mixing the stage name into the hashed payload (not just a filename
    prefix) ensures two stages given otherwise-identical inputs can never
    collide.
    """
    payload = json.dumps({"stage": stage, "input": input_digest, "config": cfg_digest})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_extract_run_id(
    file_list: str,
    config: Mapping[str, Any],
    storage_options: dict | None = None,
) -> str:
    """Compute the extract stage's run id.

    Args:
        file_list: fsspec-compatible URI to the newline-delimited listing.
        config: output-affecting extract configuration.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        16-char id over the file listing's content digest + ``config_digest(config)``.
    """
    input_digest = content_digest(file_list, storage_options)
    return _stage_run_id("extract", input_digest, config_digest(config))


def compute_assign_run_id(
    manifests_dir: str,
    config: Mapping[str, Any],
    storage_options: dict | None = None,
) -> str:
    """Compute the assign-split stage's run id.

    Args:
        manifests_dir: folder of extract manifests.
        config: output-affecting assign-split configuration.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        16-char id over the digest of the folder's ``extract-``-prefixed
        manifests + ``config_digest(config)``. Files the extract stage did not
        write — including this stage's own previous output — are ignored.
    """
    input_digest = directory_digest(
        manifests_dir,
        suffix=EXTRACT_MANIFEST_SUFFIX,
        storage_options=storage_options,
        prefix=EXTRACT_MANIFEST_PREFIX,
    )
    return _stage_run_id("assign", input_digest, config_digest(config))


def compute_build_run_id(
    split_manifest_path: str,
    config: Mapping[str, Any],
    storage_options: dict | None = None,
) -> str:
    """Compute the build stage's run id.

    Args:
        split_manifest_path: path to the split manifest.
        config: output-affecting build configuration.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        16-char id over the split manifest's content digest + ``config_digest(config)``.
    """
    input_digest = content_digest(split_manifest_path, storage_options)
    return _stage_run_id("build", input_digest, config_digest(config))
