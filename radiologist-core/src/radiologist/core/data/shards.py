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

import re
from typing import Dict, List

import fsspec  # type: ignore[import-untyped]
from braceexpand import braceexpand  # type: ignore[import-untyped]

from radiologist.utils import pathjoin

_WILDCARD_RE = re.compile(r"[*?\[]")


def _label_from_path(path: str) -> str:
    """Return the parent-directory segment of a shard path.

    Args:
        path: A POSIX path, Windows path, or cloud URL like
            s3://bucket/split/label/shard.tar.

    Returns:
        The parent directory name (the label segment).
    """
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return parts[-2]


def _split_from_path(path: str) -> str:
    """Return the grandparent-directory segment of a shard path (the split name).

    Args:
        path: A POSIX path, Windows path, or cloud URL like
            .../split/label/shard.tar.

    Returns:
        The grandparent directory name (the split segment).
    """
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return parts[-3]


def _group_by_label(paths: List[str]) -> Dict[str, List[str]]:
    by_label: Dict[str, List[str]] = {}
    for p in paths:
        by_label.setdefault(_label_from_path(p), []).append(p)
    return by_label


def _discover_shards(
    shard_root: str, splits: List[str], storage_options: dict | None = None
) -> Dict[str, Dict[str, List[str]]]:
    """Discover tar shards under {shard_root}/{split}/{label}/*.tar.

    Args:
        shard_root: Root directory or brace/wildcard spec. When shard_root
            contains '{' or a wildcard character, it is treated as a full spec
            (power mode): all paths are expanded once and routed to their split
            by reading the split segment from the path. Otherwise, discovery
            uses {shard_root}/{split}/*/*.tar (default mode).
        splits: Required split names (e.g. ["train", "val"]).

    Returns:
        Nested dict {split: {label: [tar_paths]}}.

    Raises:
        FileNotFoundError: If a required split has no matching shards.
    """
    opts = storage_options or {}
    result: Dict[str, Dict[str, List[str]]] = {}

    fs, path = fsspec.url_to_fs(shard_root, **opts)
    remote = "local" not in fs.protocol

    all_paths = []

    braceexpanded = list(braceexpand(shard_root))
    istarfile = shard_root.endswith(".tar")

    if _WILDCARD_RE.search(shard_root):
        for p in braceexpanded:
            all_paths.extend(fs.glob(p))
    else:
        for p in braceexpanded:
            if not istarfile:
                for split in splits:
                    pattern = pathjoin(p, split, "*", "*.tar")
                    all_paths.extend(fs.glob(pattern))
            else:
                all_paths = braceexpanded

    for split in splits:
        split_paths = [
            (fs.unstrip_protocol(p) if remote else p)
            for p in sorted(all_paths)
            if _split_from_path(p) == split
        ]
        if not split_paths:
            raise FileNotFoundError(
                f"No shards found for split '{split}' under '{shard_root}'"
            )
        result[split] = _group_by_label(split_paths)

    return result
