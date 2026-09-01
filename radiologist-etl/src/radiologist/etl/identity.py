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

from collections.abc import Mapping
from typing import Any


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
    raise NotImplementedError


def config_digest(config: Mapping[str, Any]) -> str:
    """Hash a config mapping via canonical (sort_keys) JSON.

    Args:
        config: mapping of output-affecting configuration values.

    Returns:
        The full 64-char hex digest; key order never affects the result.
    """
    raise NotImplementedError


def directory_digest(
    directory: str,
    suffix: str = ".jsonl",
    storage_options: dict | None = None,
) -> str:
    """Hash the sorted (name, size) pairs of a directory's matching entries.

    Args:
        directory: fsspec-compatible URI to the directory.
        suffix: only entries ending in this suffix are included.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        The full 64-char hex digest, obtained with a single detailed listing
        call — never one stat call per entry.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
        16-char id over ``directory_digest(manifests_dir)`` + ``config_digest(config)``.
    """
    raise NotImplementedError


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
    raise NotImplementedError
