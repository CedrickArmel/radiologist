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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

_CONFIGS_DIR = Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"


def test_onnx_export_callback_importable_from_public_api():
    from radiologist.core import OnnxExportCallback

    assert OnnxExportCallback is not None


def test_onnx_export_callback_init_raises_not_implemented():
    from radiologist.core import OnnxExportCallback

    with pytest.raises(NotImplementedError):
        OnnxExportCallback(
            input_shape=(1, 3, 224, 224),
            classes=["normal", "abnormal"],
            cam_target_layer="layer4",
        )


def test_resolve_resume_ckpt_raises_not_implemented():
    from radiologist.core.resume import resolve_resume_ckpt

    cfg = OmegaConf.create({"ckpt_path": None, "resume_ref": None, "resume_path": None})
    with pytest.raises(NotImplementedError):
        resolve_resume_ckpt(cfg)


def test_restore_precision_raises_not_implemented(ckpt_path):
    from radiologist.core.resume import restore_precision

    cfg = OmegaConf.create({"trainer": {"precision": 32}})
    with pytest.raises(NotImplementedError):
        restore_precision(cfg, ckpt_path)


def test_train_yaml_declares_resume_ref_and_resume_path_null():
    cfg = OmegaConf.load(_CONFIGS_DIR / "train.yaml")
    assert cfg.resume_ref is None
    assert cfg.resume_path is None
    assert cfg.ckpt_path is None


def test_onnx_export_config_exists_and_not_wired_into_default_callbacks():
    onnx_export_yaml = _CONFIGS_DIR / "callbacks" / "onnx_export.yaml"
    default_yaml = _CONFIGS_DIR / "callbacks" / "default.yaml"

    assert onnx_export_yaml.exists()

    cfg = OmegaConf.load(onnx_export_yaml)
    assert cfg.onnx_export._target_ == "radiologist.core.OnnxExportCallback"
    assert cfg.onnx_export.opset == 18

    assert "onnx_export" not in default_yaml.read_text()
