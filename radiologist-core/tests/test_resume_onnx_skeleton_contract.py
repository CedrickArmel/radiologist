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

from omegaconf import OmegaConf

_CONFIGS_DIR = Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"


def test_onnx_export_callback_importable_from_public_api():
    from radiologist.core import OnnxExportCallback

    assert OnnxExportCallback is not None


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
