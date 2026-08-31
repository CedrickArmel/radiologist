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

"""Shared fixtures for radiologist-cli tests.

``build_det_onnx``/``build_mcd_onnx`` are copied from
``radiologist-inference/radiologist_inference_tests/_helpers.py``, and the
tiny real WebDataset shards / ``LModule`` net / checkpoint fixtures are
replicated from ``radiologist-core/radiologist_core_tests/conftest.py`` --
cross-package test-directory imports are not supported in this repo's
layout, so these builders are duplicated here rather than imported.
"""

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest
from omegaconf import DictConfig, OmegaConf
from PIL import Image as PILImage

CLASSES = ["NORMAL", "ABNORMAL"]
INPUT_SHAPE = [1, 3, 224, 224]
N_CLASSES = len(CLASSES)
N_FEATURES = 3 * 224 * 224


def _add_metadata(
    model: onnx.ModelProto,
    extra: Optional[dict] = None,
    omit_keys: Optional[List[str]] = None,
) -> onnx.ModelProto:
    base = {
        "classes": json.dumps(CLASSES),
        "input_shape": json.dumps(INPUT_SHAPE),
        "cam_target_layer": "features.28",
        "output_names": json.dumps(["logits", "feature_maps"]),
    }
    if extra:
        base.update(extra)
    if omit_keys:
        for key in omit_keys:
            base.pop(key, None)
    del model.metadata_props[:]
    for k, v in base.items():
        e = model.metadata_props.add()
        e.key = k
        e.value = v
    return model


def _build_det_onnx(
    tmp_path,
    priors: Optional[dict] = None,
    filename: str = "model_det.onnx",
    feat_nonzero: bool = False,
    omit_keys: Optional[List[str]] = None,
) -> str:
    """Build a minimal deterministic 2-class ONNX classifier for tests.

    Outputs: logits (1, N_CLASSES) via Softmax, feature_maps (1, 64, 7, 7)
    constant. Embeds required metadata keys. Optional training_prior
    embedded when priors given. When feat_nonzero=True, feature_maps filled
    with a deterministic non-zero constant.
    """
    np.random.seed(42)
    W = np.random.randn(N_CLASSES, N_FEATURES).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)

    W_init = onh.from_array(W, name="W")
    b_init = onh.from_array(b, name="b")
    feat_value = (
        np.ones((1, 64, 7, 7), dtype=np.float32) * 0.5
        if feat_nonzero
        else np.zeros((1, 64, 7, 7), dtype=np.float32)
    )
    feat_const = onh.from_array(feat_value, name="feat_const")
    shape_data = onh.from_array(
        np.array([1, N_FEATURES], dtype=np.int64), name="reshape_shape"
    )

    FLOAT = onnx.TensorProto.FLOAT
    logits_vi = oh.make_tensor_value_info("logits", FLOAT, [1, N_CLASSES])
    feature_maps_vi = oh.make_tensor_value_info("feature_maps", FLOAT, [1, 64, 7, 7])

    graph = oh.make_graph(
        nodes=[
            oh.make_node(
                "Reshape", inputs=["input", "reshape_shape"], outputs=["reshape_out"]
            ),
            oh.make_node(
                "Gemm",
                inputs=["reshape_out", "W", "b"],
                outputs=["gemm_out"],
                transB=1,
            ),
            oh.make_node("Softmax", inputs=["gemm_out"], outputs=["logits"], axis=1),
            oh.make_node("Identity", inputs=["feat_const"], outputs=["feature_maps"]),
        ],
        name="det_classifier",
        inputs=[oh.make_tensor_value_info("input", FLOAT, INPUT_SHAPE)],
        outputs=[logits_vi, feature_maps_vi],
        initializer=[W_init, b_init, shape_data, feat_const],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8

    extra = {}
    if priors is not None:
        extra["training_prior"] = json.dumps(priors)
    _add_metadata(model, extra, omit_keys=omit_keys)

    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


def _build_mcd_onnx(tmp_path, filename: str = "model_mcd.onnx") -> str:
    """Build a stochastic MCD ONNX model whose logits vary each forward pass.

    Uses RandomUniform so each session.run() produces different logits.
    """
    FLOAT = onnx.TensorProto.FLOAT
    logits_vi = oh.make_tensor_value_info("logits", FLOAT, [1, N_CLASSES])

    graph = oh.make_graph(
        nodes=[
            oh.make_node(
                "RandomUniform",
                inputs=[],
                outputs=["rand_out"],
                dtype=1,
                shape=[1, N_CLASSES],
            ),
            oh.make_node("Softmax", inputs=["rand_out"], outputs=["logits"], axis=1),
        ],
        name="mcd_classifier",
        inputs=[oh.make_tensor_value_info("input", FLOAT, INPUT_SHAPE)],
        outputs=[logits_vi],
        initializer=[],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    _add_metadata(model)

    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


def _make_png_path(tmp_path, width: int = 64, height: int = 64) -> str:
    """Write a minimal RGB PNG to tmp_path and return its path as a string."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    path = tmp_path / "input.png"
    img.save(path, format="PNG")
    return str(path)


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def build_det_onnx():
    """Factory fixture building a minimal deterministic 2-class ONNX model."""
    return _build_det_onnx


@pytest.fixture
def build_mcd_onnx():
    """Factory fixture building a stochastic MC-Dropout ONNX model."""
    return _build_mcd_onnx


@pytest.fixture
def make_png_path():
    """Factory fixture writing a minimal RGB PNG and returning its path."""
    return _make_png_path


@pytest.fixture
def make_png_bytes():
    """Factory fixture returning minimal RGB PNG bytes."""
    return _make_png_bytes


def _make_shard_png_bytes() -> bytes:
    """Create a minimal valid 4x4 RGB PNG image as bytes via PIL."""
    img = PILImage.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_shard(path: Path, keys_and_labels: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = _make_shard_png_bytes()
    with tarfile.open(str(path), "w") as tf:
        for key, label in keys_and_labels:
            for ext, data in [("png", png_bytes), ("cls", label.encode())]:
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                buf = io.BytesIO(data)
                info.size = len(data)
                tf.addfile(info, buf)


@pytest.fixture()
def shard_root(tmp_path: Path) -> Path:
    """Minimal shard tree with train/val splits and two classes."""
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
    }
    for (split, label), items in samples.items():
        shard_path = root / split / label / f"{split}-{label.lower()}-000000.tar"
        _write_shard(shard_path, items)
    return root


@pytest.fixture()
def split_manifest_uri(tmp_path: Path, shard_root: Path) -> str:
    """JSONL manifest with one record per sample in shard_root."""
    manifest_path = tmp_path / "manifest.jsonl"
    records = []
    splits_labels = [
        ("train", "NORMAL"),
        ("train", "ABNORMAL"),
        ("val", "NORMAL"),
        ("val", "ABNORMAL"),
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


_LMODULE_NET_CFG: Dict[str, Any] = {
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

# Dropout layer index in the net above -- used by OnnxExportCallback's CAM hook.
CAM_TARGET_LAYER = "2"

_LMODULE_CFG: Dict[str, Any] = {
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
    from radiologist.core import LModule

    return LModule(cfg=OmegaConf.create(_LMODULE_CFG))


@pytest.fixture()
def ckpt_path(tmp_path: Path, lmodule) -> str:
    """Real Lightning checkpoint loadable via ``LModule.load_from_checkpoint``."""
    import lightning as L
    import torch

    path = str(tmp_path / "test.ckpt")
    ckpt = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": L.__version__,
        "state_dict": lmodule.state_dict(),
        "hyper_parameters": {"cfg": _LMODULE_CFG},
        "precision": "32-true",
    }
    torch.save(ckpt, path)
    return path


def _datamodule_cfg(shard_root: Path, split_manifest_uri: str, batch_size: int = 2):
    transform = {
        "_target_": "torchvision.transforms.Compose",
        "transforms": [
            {"_target_": "torchvision.transforms.Resize", "size": [8, 8]},
            {"_target_": "torchvision.transforms.ToTensor"},
        ],
    }
    loader_partial = {
        "_target_": "webdataset.WebLoader",
        "_partial_": True,
        "batch_size": None,
        "num_workers": 0,
    }
    return {
        "_target_": "radiologist.core.WebDatasetDataModule",
        "shard_root": str(shard_root),
        "split_manifest_uri": split_manifest_uri,
        "label_map": {"NORMAL": "normal", "ABNORMAL": "abnormal"},
        "train_transform": transform,
        "eval_transform": transform,
        "train_loader": loader_partial,
        "eval_loader": loader_partial,
        "batch_size": batch_size,
        "classes": ["abnormal", "normal"],
    }


@pytest.fixture()
def base_cfg(tmp_path: Path, shard_root: Path, split_manifest_uri: str) -> DictConfig:
    """Full, real, minimal training config -- train stage only, no test stage.

    ``train()`` is exercised with real (tiny) Lightning components: a real
    ``WebDatasetDataModule`` over on-disk tar shards, a real ``LModule`` with
    a two-layer conv net, and a real ``Trainer``. No mocks of owned code.
    """
    cfg = {
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "resume_ref": None,
        "resume_path": None,
        "optimized_metric": None,
        "paths": {"output_dir": str(tmp_path / "out")},
        "module": _LMODULE_CFG,
        "datamodule": _datamodule_cfg(shard_root, split_manifest_uri),
        "trainer": {
            "_target_": "lightning.pytorch.Trainer",
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "default_root_dir": str(tmp_path / "out"),
            "use_distributed_sampler": False,
        },
        "callbacks": {
            "model_checkpoint": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "dirpath": str(tmp_path / "out" / "checkpoints"),
                "monitor": "val_score",
                "mode": "max",
                "save_top_k": 1,
                "save_last": True,
            },
            "onnx_export": {
                "_target_": "radiologist.core.OnnxExportCallback",
                "input_shape": [1, 3, 8, 8],
                "classes": ["healthy", "sick"],
                "cam_target_layer": CAM_TARGET_LAYER,
            },
        },
        "loggers": None,
    }
    return OmegaConf.create(cfg)
