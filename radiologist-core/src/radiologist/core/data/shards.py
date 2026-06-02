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
import tarfile
from typing import Callable, Dict, List


def _discover_shards(
    shard_root: str, splits: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """Discover tar shards under {shard_root}/{split}/{label}/*.tar.

    Args:
        shard_root: Root directory containing split sub-directories.
        splits: Required split names (e.g. ["train", "val"]).

    Returns:
        Nested dict {split: {label: [tar_paths]}}.

    Raises:
        FileNotFoundError: If a required split directory has no shards.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    for split in splits:
        pattern = f"{shard_root}/{split}/*/*.tar"
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No shards found for split '{split}' under '{shard_root}/{split}'"
            )
        by_label: Dict[str, List[str]] = {}
        for p in paths:
            label = p.split("/")[-2]
            by_label.setdefault(label, []).append(p)
        result[split] = by_label
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
