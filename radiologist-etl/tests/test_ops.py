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

import numpy as np

from radiologist.etl.ops import lung_out_of_frame


def test_returns_true_when_nonzero_pixel_on_first_row() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 5] = 1
    assert lung_out_of_frame(mask) is True


def test_returns_true_when_nonzero_pixel_on_last_row() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[9, 5] = 1
    assert lung_out_of_frame(mask) is True


def test_returns_true_when_nonzero_pixel_on_first_column() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 0] = 1
    assert lung_out_of_frame(mask) is True


def test_returns_true_when_nonzero_pixel_on_last_column() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 9] = 1
    assert lung_out_of_frame(mask) is True


def test_returns_false_for_all_zero_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert lung_out_of_frame(mask) is False


def test_returns_false_when_nonzero_only_in_interior() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3, 4] = 1
    mask[5, 6] = 255
    assert lung_out_of_frame(mask) is False


def test_collapses_3channel_mask_before_checking_borders() -> None:
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0, 5, 2] = 1
    assert lung_out_of_frame(mask) is True


def test_returns_false_for_3channel_mask_with_nonzero_only_in_interior() -> None:
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[3, 4, 0] = 1
    assert lung_out_of_frame(mask) is False
