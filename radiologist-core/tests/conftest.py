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
import tarfile
from functools import partial
from pathlib import Path

import pytest
import torchvision.transforms as T  # type: ignore[import-untyped]
import webdataset as wds  # type: ignore[import-untyped]


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


@pytest.fixture(scope="session")
def label_map() -> dict:
    return {"NORMAL": "normal", "ABNORMAL": "abnormal"}


@pytest.fixture(scope="session")
def classes() -> list:
    return ["abnormal", "normal"]


@pytest.fixture(scope="session")
def transform() -> T.Compose:
    return T.Compose([T.Resize((8, 8)), T.ToTensor()])


@pytest.fixture(scope="session")
def train_transform(transform: T.Compose) -> T.Compose:
    return transform


@pytest.fixture(scope="session")
def eval_transform(transform: T.Compose) -> T.Compose:
    return transform


@pytest.fixture(scope="session")
def train_loader_partial() -> partial:
    return partial(wds.WebLoader, batch_size=None, num_workers=0)


@pytest.fixture(scope="session")
def eval_loader_partial() -> partial:
    return partial(wds.WebLoader, batch_size=None, num_workers=0)


@pytest.fixture(scope="session")
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


@pytest.fixture()
def dm(
    split_manifest_uri,
    transform,
    label_map,
    classes,
    train_loader_partial,
    eval_loader_partial,
    batch_size,
    shard_root,
):
    from radiologist.core import WebDatasetDataModule

    return WebDatasetDataModule(
        shard_root=str(shard_root),
        split_manifest_uri=split_manifest_uri,
        label_map=label_map,
        train_transform=transform,
        eval_transform=transform,
        train_loader=train_loader_partial,
        eval_loader=eval_loader_partial,
        batch_size=batch_size,
        classes=classes,
    )


_LMODULE_NET_CFG = {
    "_target_": "torch.nn.Sequential",
    "_args_": [
        {
            "_target_": "torch.nn.Conv2d",
            "in_channels": 3,
            "out_channels": 4,
            "kernel_size": 3,
            "padding": 1,
        },
        {"_target_": "torch.nn.ReLU"},
        {"_target_": "torch.nn.Dropout", "p": 0.5},
        {"_target_": "torch.nn.AdaptiveAvgPool2d", "output_size": [1, 1]},
        {"_target_": "torch.nn.Flatten"},
        {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 2},
    ],
}

_LMODULE_CFG = {
    "net": _LMODULE_NET_CFG,
    "loss": {"_target_": "radiologist.core.FocalLoss"},
    "metric": {
        "_target_": "torchmetrics.classification.MulticlassFBetaScore",
        "_partial_": True,
        "beta": 1.0,
        "num_classes": 2,
    },
    "optimizer": {
        "_target_": "torch.optim.Adam",
        "_partial_": True,
        "lr": 1e-3,
    },
    "scheduler": None,
    "trainable_layers": None,
    "priors": None,
}


@pytest.fixture()
def lmodule():
    from omegaconf import OmegaConf

    from radiologist.core import LModule

    return LModule(cfg=OmegaConf.create(_LMODULE_CFG))


@pytest.fixture()
def ckpt_path(tmp_path, lmodule):
    """Real Lightning checkpoint loadable via LModule.load_from_checkpoint.

    Stores the cfg DictConfig in hyper_parameters so Lightning reconstructs
    the module via LModule(cfg=...) using the same architecture as lmodule.
    """
    import lightning as L
    import torch

    path = str(tmp_path / "test.ckpt")
    ckpt = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": L.__version__,
        "state_dict": lmodule.state_dict(),
        "hyper_parameters": {"cfg": _LMODULE_CFG},
    }
    torch.save(ckpt, path)
    return path
