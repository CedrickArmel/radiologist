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

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _helpers import build_det_onnx, build_mcd_onnx  # noqa: E402


@pytest.fixture()
def det_onnx_path(tmp_path):
    return build_det_onnx(tmp_path)


@pytest.fixture()
def det_onnx_path_nonzero(tmp_path):
    return build_det_onnx(tmp_path, feat_nonzero=True)


@pytest.fixture()
def mcd_onnx_path(tmp_path):
    return build_mcd_onnx(tmp_path)


@pytest.fixture()
def predictor_with_mcd(tmp_path):
    from radiologist.inference.mc_dropout import MCDropoutPredictor

    det = build_det_onnx(tmp_path, filename="det.onnx")
    mcd = build_mcd_onnx(tmp_path, filename="mcd.onnx")
    return MCDropoutPredictor.from_path(det_path=det, mcd_path=mcd)


@pytest.fixture()
def predictor_without_mcd(tmp_path):
    from radiologist.inference.mc_dropout import MCDropoutPredictor

    det = build_det_onnx(tmp_path, filename="det_only.onnx")
    return MCDropoutPredictor.from_path(det_path=det)


@pytest.fixture()
def sample_image():
    return np.zeros((224, 224, 3), dtype=np.uint8)
