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

import io
import json
import sys
import tarfile
from functools import partial
from pathlib import Path

import pytest
import torchvision.transforms as T  # type: ignore[import-untyped]
import webdataset as wds  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _make_png_bytes() -> bytes:
    """Create a minimal valid 4x4 RGB PNG image as bytes via PIL."""
    import io as _io

    from PIL import Image  # type: ignore[import-untyped]

    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_shard(path: Path, keys_and_labels: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = _make_png_bytes()
    with tarfile.open(str(path), "w") as tf:
        for key, label in keys_and_labels:
            for ext, data in [("png", png_bytes), ("cls", label.encode())]:
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                buf = io.BytesIO(data)
                info.size = len(data)
                tf.addfile(info, buf)


@pytest.fixture()
def shard_root(tmp_path: Path) -> Path:
    """Minimal shard tree with train/val/test splits and two classes."""
    root = tmp_path / "shards"
    samples = {
        ("train", "NORMAL"): [
            ("train-normal-000000", "NORMAL"),
            ("train-normal-000001", "NORMAL"),
        ],
        ("train", "ABNORMAL"): [
            ("train-abnormal-000000", "ABNORMAL"),
            ("train-abnormal-000001", "ABNORMAL"),
        ],
        ("val", "NORMAL"): [
            ("val-normal-000000", "NORMAL"),
            ("val-normal-000001", "NORMAL"),
        ],
        ("val", "ABNORMAL"): [
            ("val-abnormal-000000", "ABNORMAL"),
            ("val-abnormal-000001", "ABNORMAL"),
        ],
        ("test", "NORMAL"): [
            ("test-normal-000000", "NORMAL"),
            ("test-normal-000001", "NORMAL"),
        ],
        ("test", "ABNORMAL"): [
            ("test-abnormal-000000", "ABNORMAL"),
            ("test-abnormal-000001", "ABNORMAL"),
        ],
    }
    for (split, label), items in samples.items():
        shard_path = root / split / label / f"{split}-{label.lower()}-000000.tar"
        _write_shard(shard_path, items)
    return root


@pytest.fixture()
def label_map() -> dict:
    return {"NORMAL": "normal", "ABNORMAL": "abnormal"}


@pytest.fixture()
def classes() -> list:
    return ["abnormal", "normal"]


@pytest.fixture()
def train_transform() -> T.Compose:
    return T.Compose([T.Resize((8, 8)), T.ToTensor()])


@pytest.fixture()
def eval_transform() -> T.Compose:
    return T.Compose([T.Resize((8, 8)), T.ToTensor()])


@pytest.fixture()
def train_loader_partial() -> partial:
    return partial(wds.WebLoader, batch_size=2, num_workers=0)


@pytest.fixture()
def eval_loader_partial() -> partial:
    return partial(wds.WebLoader, batch_size=2, num_workers=0)


@pytest.fixture()
def batch_size() -> int:
    return 2


@pytest.fixture()
def split_manifest_uri(tmp_path: Path, shard_root: Path) -> str:
    """JSONL manifest with one record per sample in shard_root (12 records total)."""
    manifest_path = tmp_path / "manifest.jsonl"
    records = []
    splits_labels = [
        ("train", "NORMAL"),
        ("train", "ABNORMAL"),
        ("val", "NORMAL"),
        ("val", "ABNORMAL"),
        ("test", "NORMAL"),
        ("test", "ABNORMAL"),
    ]
    for split, label in splits_labels:
        shard_rel = f"{split}/{label}/{split}-{label.lower()}-000000.tar"
        for i in range(2):
            records.append(
                {
                    "manifest_id": "test0000000000000000",
                    "path": f"s3://fake/{split}/{label}/img_{i}.png",
                    "filename": f"img_{i}.png",
                    "label": label,
                    "split": split,
                    "shard": shard_rel,
                    "lung_out_of_frame": None,
                    "excluded": False,
                    "exclusion_reason": "",
                }
            )
    with open(str(manifest_path), "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(manifest_path)
