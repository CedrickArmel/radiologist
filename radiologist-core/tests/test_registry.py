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

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from radiologist.core import LModule

import pytest
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassFBetaScore

# ---------------------------------------------------------------------------
# Helpers — minimal net and LModule
# ---------------------------------------------------------------------------


def _make_tiny_net() -> nn.Sequential:
    """Tiny net with a Dropout so MCD export can find it."""
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def _make_lmodule(net: nn.Module) -> "LModule":
    from radiologist.core import FocalLoss, LModule

    return LModule(
        net=net,
        loss=FocalLoss(),
        metric=partial(MulticlassFBetaScore, beta=1.0, num_classes=2),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )


# ---------------------------------------------------------------------------
# Wandb stub factory
# ---------------------------------------------------------------------------


def _make_fake_wandb_api(tmp_path: Path, run_id: str, precision: str) -> MagicMock:
    """Return a wandb.Api() mock that serves a fake artifact backed by tmp_path."""
    artifact_mock = MagicMock()
    artifact_mock.download.return_value = str(tmp_path)

    source_run = MagicMock()
    source_run.id = run_id
    source_run.config = {"trainer": {"precision": precision}}
    artifact_mock.logged_by.return_value = source_run

    api_mock = MagicMock()
    api_mock.artifact.return_value = artifact_mock
    return api_mock


def _make_fake_wandb_module(tmp_path: Path, run_id: str, precision: str) -> MagicMock:
    """Return a wandb module stub usable in patch.dict(sys.modules, ...)."""
    fake = MagicMock()
    fake.Api.return_value = _make_fake_wandb_api(tmp_path, run_id, precision)

    linked_artifact = MagicMock()
    linked_artifact.qualified_name = f"wandb-registry-model/{run_id}:latest"

    artifact_log_mock = MagicMock()
    artifact_log_mock.qualified_name = f"wandb-registry-model/{run_id}:latest"
    artifact_log_mock.wait.return_value = None

    run_mock = MagicMock()
    run_mock.__enter__ = MagicMock(return_value=run_mock)
    run_mock.__exit__ = MagicMock(return_value=False)
    run_mock.log_artifact.return_value = artifact_log_mock
    fake.init.return_value = run_mock
    fake.Artifact.return_value = artifact_log_mock

    return fake


# ---------------------------------------------------------------------------
# AC: pull_checkpoint — RuntimeError when wandb not installed
# ---------------------------------------------------------------------------


def test_pull_checkpoint_raises_runtime_error_when_wandb_absent():
    import radiologist.core.registry.pull as pull_mod

    with patch.object(pull_mod, "wandb", None):
        with pytest.raises(RuntimeError, match="wandb"):
            pull_mod.pull_checkpoint("entity/project/model:best", "/tmp/ckpt")


# ---------------------------------------------------------------------------
# AC: pull_checkpoint — FileNotFoundError when no .ckpt in artifact dir
# ---------------------------------------------------------------------------


def test_pull_checkpoint_raises_file_not_found_when_no_ckpt(tmp_path):
    import radiologist.core.registry.pull as pull_mod

    fake_api = MagicMock()
    artifact_mock = MagicMock()
    artifact_mock.download.return_value = str(tmp_path)  # empty — no .ckpt
    fake_api.artifact.return_value = artifact_mock
    fake_wandb = MagicMock()
    fake_wandb.Api.return_value = fake_api

    with patch.object(pull_mod, "wandb", fake_wandb):
        with pytest.raises(FileNotFoundError):
            pull_mod.pull_checkpoint("entity/project/model:best", str(tmp_path))


# ---------------------------------------------------------------------------
# AC: pull_checkpoint — returns path when .ckpt exists
# ---------------------------------------------------------------------------


def test_pull_checkpoint_returns_ckpt_path_when_ckpt_exists(tmp_path):
    import radiologist.core.registry.pull as pull_mod

    ckpt_file = tmp_path / "model.ckpt"
    ckpt_file.write_text("dummy")

    fake_api = MagicMock()
    artifact_mock = MagicMock()
    artifact_mock.download.return_value = str(tmp_path)
    fake_api.artifact.return_value = artifact_mock
    fake_wandb = MagicMock()
    fake_wandb.Api.return_value = fake_api

    with patch.object(pull_mod, "wandb", fake_wandb):
        result = pull_mod.pull_checkpoint("entity/project/model:best", str(tmp_path))

    assert result == str(ckpt_file)


# ---------------------------------------------------------------------------
# AC: promote_to_registry — RuntimeError when wandb absent
# ---------------------------------------------------------------------------


def test_promote_to_registry_raises_runtime_error_when_wandb_absent():
    import radiologist.core.registry.promote as promo_mod

    with patch.object(promo_mod, "wandb", None):
        with pytest.raises(RuntimeError, match="wandb"):
            promo_mod.promote_to_registry(
                artifact="entity/project/model:best",
                collection="my-collection",
                registry_alias="latest",
                input_shape=(1, 3, 8, 8),
                classes=["healthy", "sick"],
                cam_target_layer="0",
                local_dir="/tmp",
            )


# ---------------------------------------------------------------------------
# AC: promote_to_registry — RuntimeError when onnx absent
# ---------------------------------------------------------------------------


def test_promote_to_registry_raises_runtime_error_when_onnx_absent(tmp_path):
    import radiologist.core.registry.promote as promo_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)
    fake_wandb = _make_fake_wandb_module(tmp_path, "abc123", "16-mixed")
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.write_text("placeholder")

    with (
        patch.object(promo_mod, "wandb", fake_wandb),
        patch.object(promo_mod, "onnx", None),
        patch.object(promo_mod, "pull_checkpoint", return_value=str(ckpt_path)),
        patch.object(promo_mod.LModule, "load_from_checkpoint", return_value=lm),
    ):
        with pytest.raises(RuntimeError, match="onnx"):
            promo_mod.promote_to_registry(
                artifact="entity/project/model:best",
                collection="my-collection",
                registry_alias="latest",
                input_shape=(1, 3, 8, 8),
                classes=["healthy", "sick"],
                cam_target_layer="0",
                local_dir=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# AC: promote_to_registry — AttributeError for bad cam_target_layer
# ---------------------------------------------------------------------------


def test_promote_to_registry_raises_attribute_error_for_bad_cam_layer(tmp_path):
    import radiologist.core.registry.promote as promo_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)
    fake_wandb = _make_fake_wandb_module(tmp_path, "abc123", "16-mixed")
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.write_text("placeholder")

    with (
        patch.object(promo_mod, "wandb", fake_wandb),
        patch.object(promo_mod, "pull_checkpoint", return_value=str(ckpt_path)),
        patch.object(promo_mod.LModule, "load_from_checkpoint", return_value=lm),
    ):
        with pytest.raises(AttributeError):
            promo_mod.promote_to_registry(
                artifact="entity/project/model:best",
                collection="my-collection",
                registry_alias="latest",
                input_shape=(1, 3, 8, 8),
                classes=["healthy", "sick"],
                cam_target_layer="nonexistent.deep.layer",
                local_dir=str(tmp_path),
            )

    assert list(tmp_path.glob("*.onnx")) == []


# ---------------------------------------------------------------------------
# Fixture: promote_result — both ONNX files on disk
# ---------------------------------------------------------------------------


@pytest.fixture()
def promote_result(tmp_path):
    """Run promote_to_registry with full stubs; return (local_dir, run_id, result)."""
    import radiologist.core.registry.promote as promo_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)
    run_id = "abc123"
    precision = "16-mixed"
    fake_wandb = _make_fake_wandb_module(tmp_path, run_id, precision)
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.write_text("placeholder")

    with (
        patch.object(promo_mod, "wandb", fake_wandb),
        patch.object(promo_mod, "pull_checkpoint", return_value=str(ckpt_path)),
        patch.object(promo_mod.LModule, "load_from_checkpoint", return_value=lm),
    ):
        result = promo_mod.promote_to_registry(
            artifact="entity/project/model:best",
            collection="my-collection",
            registry_alias="latest",
            input_shape=(1, 3, 8, 8),
            classes=["healthy", "sick"],
            cam_target_layer="2",  # Sequential index — Dropout layer
            local_dir=str(tmp_path),
        )

    return tmp_path, run_id, result


# ---------------------------------------------------------------------------
# AC: both ONNX files created
# ---------------------------------------------------------------------------


def test_promote_creates_deterministic_and_mcd_onnx_files(promote_result):
    local_dir, run_id, _ = promote_result
    assert (local_dir / f"model-{run_id}.onnx").exists()
    assert (local_dir / f"model-{run_id}-mcd.onnx").exists()


# ---------------------------------------------------------------------------
# AC: deterministic ONNX metadata_props
# ---------------------------------------------------------------------------


def test_deterministic_onnx_has_required_metadata_keys(promote_result):
    import onnx

    local_dir, run_id, _ = promote_result
    model = onnx.load(str(local_dir / f"model-{run_id}.onnx"))
    props = {p.key: p.value for p in model.metadata_props}

    for key in ("precision", "run_id", "input_shape", "classes", "framework"):
        assert key in props, f"Missing metadata key: {key}"

    assert "cam_target_layer" in props
    output_names = json.loads(props["output_names"])
    assert "logits" in output_names
    assert "feature_maps" in output_names


# ---------------------------------------------------------------------------
# AC: MCD ONNX metadata_props
# ---------------------------------------------------------------------------


def test_mcd_onnx_has_required_metadata_keys_and_mc_dropout_true(promote_result):
    import onnx

    local_dir, run_id, _ = promote_result
    model = onnx.load(str(local_dir / f"model-{run_id}-mcd.onnx"))
    props = {p.key: p.value for p in model.metadata_props}

    for key in ("precision", "run_id", "input_shape", "classes", "framework"):
        assert key in props, f"Missing metadata key: {key}"

    assert props.get("mc_dropout") == "true"


# ---------------------------------------------------------------------------
# AC: precision comes from W&B run config
# ---------------------------------------------------------------------------


def test_precision_metadata_comes_from_wandb_run_config(tmp_path):
    import onnx

    import radiologist.core.registry.promote as promo_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)
    custom_precision = "bf16-mixed"
    run_id = "xyz789"
    fake_wandb = _make_fake_wandb_module(tmp_path, run_id, custom_precision)
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.write_text("placeholder")

    with (
        patch.object(promo_mod, "wandb", fake_wandb),
        patch.object(promo_mod, "pull_checkpoint", return_value=str(ckpt_path)),
        patch.object(promo_mod.LModule, "load_from_checkpoint", return_value=lm),
    ):
        promo_mod.promote_to_registry(
            artifact="entity/project/model:best",
            collection="my-collection",
            registry_alias="latest",
            input_shape=(1, 3, 8, 8),
            classes=["healthy", "sick"],
            cam_target_layer="2",
            local_dir=str(tmp_path),
        )

    model = onnx.load(str(tmp_path / f"model-{run_id}.onnx"))
    props = {p.key: p.value for p in model.metadata_props}
    assert props["precision"] == custom_precision


# ---------------------------------------------------------------------------
# AC: deterministic ONNX is deterministic
# ---------------------------------------------------------------------------


def test_deterministic_onnx_gives_identical_outputs_on_two_runs(promote_result):
    import numpy as np
    import onnxruntime as ort

    local_dir, run_id, _ = promote_result
    sess = ort.InferenceSession(str(local_dir / f"model-{run_id}.onnx"))
    inp = np.random.randn(1, 3, 8, 8).astype(np.float32)
    out1 = sess.run(None, {sess.get_inputs()[0].name: inp})
    out2 = sess.run(None, {sess.get_inputs()[0].name: inp})
    np.testing.assert_array_equal(out1[0], out2[0])


# ---------------------------------------------------------------------------
# AC: MCD ONNX retains Dropout nodes
# ---------------------------------------------------------------------------


def test_mcd_onnx_retains_dropout_nodes(promote_result):
    import onnx

    local_dir, run_id, _ = promote_result
    model = onnx.load(str(local_dir / f"model-{run_id}-mcd.onnx"))
    dropout_nodes = [n for n in model.graph.node if n.op_type == "Dropout"]
    assert len(dropout_nodes) > 0, "Expected at least one Dropout node in MCD ONNX"


# ---------------------------------------------------------------------------
# AC: MCD ONNX is stochastic
# ---------------------------------------------------------------------------


def test_mcd_onnx_gives_different_outputs_on_two_runs(promote_result):
    import numpy as np
    import onnxruntime as ort

    local_dir, run_id, _ = promote_result
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(local_dir / f"model-{run_id}-mcd.onnx"),
        sess_options=so,
    )
    inp = np.random.randn(1, 3, 8, 8).astype(np.float32)

    outputs = [sess.run(None, {sess.get_inputs()[0].name: inp})[0] for _ in range(10)]
    all_same = all(np.array_equal(outputs[0], o) for o in outputs[1:])
    assert not all_same, "MCD ONNX should be stochastic but all outputs were identical"


# ---------------------------------------------------------------------------
# AC: idempotent — re-running overwrites files
# ---------------------------------------------------------------------------


def test_promote_is_idempotent_overwrites_existing_files(tmp_path):
    import radiologist.core.registry.promote as promo_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)
    run_id = "abc123"
    ckpt_path = tmp_path / "model.ckpt"
    ckpt_path.write_text("placeholder")

    kwargs = dict(
        artifact="entity/project/model:best",
        collection="my-collection",
        registry_alias="latest",
        input_shape=(1, 3, 8, 8),
        classes=["healthy", "sick"],
        cam_target_layer="2",
        local_dir=str(tmp_path),
    )

    for _ in range(2):
        fake_wandb = _make_fake_wandb_module(tmp_path, run_id, "16-mixed")
        with (
            patch.object(promo_mod, "wandb", fake_wandb),
            patch.object(promo_mod, "pull_checkpoint", return_value=str(ckpt_path)),
            patch.object(promo_mod.LModule, "load_from_checkpoint", return_value=lm),
        ):
            promo_mod.promote_to_registry(**kwargs)

    assert (tmp_path / f"model-{run_id}.onnx").exists()
    assert (tmp_path / f"model-{run_id}-mcd.onnx").exists()


# ---------------------------------------------------------------------------
# AC: promote returns linked qualified name
# ---------------------------------------------------------------------------


def test_promote_returns_qualified_name(promote_result):
    _, run_id, result = promote_result
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# AC: public API importable from radiologist.core
# ---------------------------------------------------------------------------


def test_pull_checkpoint_and_promote_importable_from_core():
    from radiologist.core import promote_to_registry, pull_checkpoint

    assert callable(pull_checkpoint)
    assert callable(promote_to_registry)
