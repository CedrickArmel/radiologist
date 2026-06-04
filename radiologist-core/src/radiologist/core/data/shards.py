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

import glob
import re
import tarfile
from typing import Callable, Dict, List

import fsspec  # type: ignore[import-untyped]
from braceexpand import braceexpand  # type: ignore[import-untyped]

try:
    import s3fs  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    pass

try:
    import gcsfs  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    pass

_WILDCARD_RE = re.compile(r"[*?\[]")


def _expand_spec(spec: str) -> List[str]:
    """Expand a shard spec to a sorted unique list of fully-qualified URLs.

    Args:
        spec: A brace-range spec, a wildcard glob, or a literal URL/path.

    Returns:
        Sorted unique list of expanded paths or URLs.
    """
    if "{" in spec:
        expanded = list(braceexpand(spec))
        wildcards = [p for p in expanded if _WILDCARD_RE.search(p)]
        literals = [p for p in expanded if not _WILDCARD_RE.search(p)]
        resolved: List[str] = list(literals)
        for pattern in wildcards:
            fs, path = fsspec.core.url_to_fs(pattern)
            resolved.extend(fs.glob(path))
        return sorted(set(resolved))
    if _WILDCARD_RE.search(spec):
        fs, path = fsspec.core.url_to_fs(spec)
        return sorted(set(fs.glob(path)))
    return [spec]


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
    shard_root: str, splits: List[str]
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
    result: Dict[str, Dict[str, List[str]]] = {}

    if "{" in shard_root or _WILDCARD_RE.search(shard_root):
        all_paths = _expand_spec(shard_root)
        for split in splits:
            split_paths = (
                [p for p in all_paths if _split_from_path(p) == split]
                if split
                else list(all_paths)
            )
            if not split_paths:
                raise FileNotFoundError(
                    f"No shards found for split '{split}' under '{shard_root}'"
                )
            result[split] = _group_by_label(split_paths)
        return result

    for split in splits:
        pattern = f"{shard_root}/{split}/*/*.tar"
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No shards found for split '{split}' under '{shard_root}/{split}'"
            )
        result[split] = _group_by_label(paths)
    return result


def _make_label_resolver(
    label_map: Dict[str, str], classes: List[str]
) -> Callable[[str], int]:
    """Build a function that maps raw ETL label strings to integer class indices.

    Args:
        label_map: Maps raw ETL label -> class name.
        classes: Ordered list of class names; index = integer target.

    Returns:
        Callable that takes a raw label string and returns its class index.

    Raises:
        KeyError: At build time if a label_map value is not in classes.
    """
    class_index: Dict[str, int] = {}
    for raw, cls_name in label_map.items():
        if cls_name not in classes:
            raise KeyError(
                f"Label map value '{cls_name}' not found in classes {classes}"
            )
        class_index[raw] = classes.index(cls_name)

    def resolve(label_str: str) -> int:
        return class_index[label_str]

    return resolve


def _count_samples(tar_path: str) -> int:
    """Count samples in a tar shard by counting '.cls' member files.

    Args:
        tar_path: Path to a WebDataset tar shard.

    Returns:
        Number of samples (one per .cls file).
    """
    count = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".cls"):
                count += 1
    return count
