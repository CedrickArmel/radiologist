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

import numpy as np
import pytest
from PIL import Image  # type: ignore[import-untyped]


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Two-class image tree: images/NORMAL/ and images/ABNORMAL/, 2 PNGs each."""
    rng = np.random.default_rng(0)
    root = tmp_path / "images"
    for label in ("NORMAL", "ABNORMAL"):
        for i in range(2):
            _write_png(
                root / label / f"img{i:03d}.png",
                rng.integers(0, 256, (10, 10, 3), dtype=np.uint8),
            )
    return root


@pytest.fixture()
def mask_dir(tmp_path: Path) -> Path:
    """Mask tree mirroring image_dir: NORMAL masks touch the border (lung out of frame),
    ABNORMAL masks are interior-only (well framed)."""
    root = tmp_path / "masks"
    border = np.zeros((10, 10, 3), dtype=np.uint8)
    border[0, :] = 255  # first row nonzero -> out of frame
    interior = np.zeros((10, 10, 3), dtype=np.uint8)
    interior[3:7, 3:7] = 255  # only interior nonzero -> well framed
    for i in range(2):
        _write_png(root / "NORMAL" / f"img{i:03d}.png", border)
        _write_png(root / "ABNORMAL" / f"img{i:03d}.png", interior)
    return root


@pytest.fixture()
def minimal_records() -> list:
    """Three ManifestRecord instances: 2 included, 1 excluded."""
    from radiologist.etl.manifest import ManifestRecord

    base = dict(manifest_id="run-test-0000001", stats={"haralick_contrast": 1.0})
    return [
        ManifestRecord(
            path="/d/NORMAL/a.png",
            filename="a.png",
            label="NORMAL",
            split="train",
            **base,
        ),
        ManifestRecord(
            path="/d/NORMAL/b.png",
            filename="b.png",
            label="NORMAL",
            split="val",
            **base,
        ),
        ManifestRecord(
            path="/d/NORMAL/c.png",
            filename="c.png",
            label="NORMAL",
            split="train",
            excluded=True,
            exclusion_reason="iqr:haralick_contrast",
            **base,
        ),
    ]
