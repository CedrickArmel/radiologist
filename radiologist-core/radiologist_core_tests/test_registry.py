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

"""Tests for core registry — issue #90: legacy APIs removed.

pull_checkpoint and promote_to_registry are removed from the public API.
export_onnx remains and its ONNX export behavior is tested here.
W&B promote behavior moved to radiologist-registry tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# AC: pull_checkpoint absent from core registry __all__
# ---------------------------------------------------------------------------


def test_pull_checkpoint_absent_from_core_registry_all():
    """pull_checkpoint must not appear in radiologist.core.registry.__all__."""
    import radiologist.core.registry as reg

    assert "pull_checkpoint" not in reg.__all__


def test_pull_checkpoint_import_raises_import_error():
    """Importing pull_checkpoint from radiologist.core.registry raises ImportError."""
    with pytest.raises(ImportError):
        from radiologist.core.registry import pull_checkpoint  # noqa: F401


def test_pull_checkpoint_absent_from_core_all():
    """pull_checkpoint must not appear in radiologist.core.__all__."""
    import radiologist.core as core

    assert "pull_checkpoint" not in core.__all__


# ---------------------------------------------------------------------------
# AC: promote_to_registry absent from core registry __all__
# ---------------------------------------------------------------------------


def test_promote_to_registry_absent_from_core_registry_all():
    """promote_to_registry must not appear in radiologist.core.registry.__all__."""
    import radiologist.core.registry as reg

    assert "promote_to_registry" not in reg.__all__


def test_promote_to_registry_import_raises_import_error():
    """Importing promote_to_registry from radiologist.core.registry raises ImportError."""
    with pytest.raises(ImportError):
        from radiologist.core.registry import promote_to_registry  # noqa: F401


def test_promote_to_registry_absent_from_core_all():
    """promote_to_registry must not appear in radiologist.core.__all__."""
    import radiologist.core as core

    assert "promote_to_registry" not in core.__all__


# ---------------------------------------------------------------------------
# AC: export_onnx remains in core registry __all__
# ---------------------------------------------------------------------------


def test_export_onnx_present_in_core_registry_all():
    """export_onnx must remain in radiologist.core.registry.__all__."""
    import radiologist.core.registry as reg

    assert "export_onnx" in reg.__all__


def test_export_onnx_importable_from_core_registry():
    """export_onnx must be importable from radiologist.core.registry."""
    from radiologist.core.registry import export_onnx  # noqa: F401

    assert callable(export_onnx)


# ---------------------------------------------------------------------------
# AC: export_onnx behavioral tests — ONNX files created with correct metadata
# ---------------------------------------------------------------------------


def test_export_onnx_creates_deterministic_and_mcd_onnx_files(ckpt_path, tmp_path):
    """export_onnx must produce det and mcd ONNX files on disk."""
    from radiologist.core.registry import export_onnx

    run_id = "abc123"
    result = export_onnx(
        ckpt_path=ckpt_path,
        run_id=run_id,
        input_shape=(1, 3, 8, 8),
        classes=["healthy", "sick"],
        cam_target_layer="2",
        out_dir=str(tmp_path),
        opset=17,
    )

    assert Path(result.det_path).exists()
    assert Path(result.mcd_path).exists()


def test_export_onnx_det_has_required_metadata(ckpt_path, tmp_path):
    """export_onnx deterministic model must embed required metadata keys."""
    import onnx

    from radiologist.core.registry import export_onnx

    run_id = "abc123"
    result = export_onnx(
        ckpt_path=ckpt_path,
        run_id=run_id,
        input_shape=(1, 3, 8, 8),
        classes=["healthy", "sick"],
        cam_target_layer="2",
        out_dir=str(tmp_path),
        opset=17,
    )

    model = onnx.load(result.det_path)
    props = {p.key: p.value for p in model.metadata_props}

    for key in ("run_id", "input_shape", "classes", "framework"):
        assert key in props, f"Missing metadata key: {key}"

    assert "cam_target_layer" in props
    output_names = json.loads(props["output_names"])
    assert "logits" in output_names
    assert "feature_maps" in output_names


def test_export_onnx_mcd_has_required_metadata(ckpt_path, tmp_path):
    """export_onnx MCD model must embed mc_dropout=true in metadata."""
    import onnx

    from radiologist.core.registry import export_onnx

    run_id = "abc123"
    result = export_onnx(
        ckpt_path=ckpt_path,
        run_id=run_id,
        input_shape=(1, 3, 8, 8),
        classes=["healthy", "sick"],
        cam_target_layer="2",
        out_dir=str(tmp_path),
        opset=17,
    )

    model = onnx.load(result.mcd_path)
    props = {p.key: p.value for p in model.metadata_props}

    for key in ("run_id", "input_shape", "classes", "framework"):
        assert key in props, f"Missing metadata key: {key}"

    assert props.get("mc_dropout") == "true"


def test_export_onnx_raises_attribute_error_for_bad_cam_layer(ckpt_path, tmp_path):
    """export_onnx raises AttributeError when cam_target_layer does not exist."""
    from radiologist.core.registry import export_onnx

    with pytest.raises(AttributeError):
        export_onnx(
            ckpt_path=ckpt_path,
            run_id="abc123",
            input_shape=(1, 3, 8, 8),
            classes=["healthy", "sick"],
            cam_target_layer="nonexistent.deep.layer",
            out_dir=str(tmp_path),
            opset=17,
        )
