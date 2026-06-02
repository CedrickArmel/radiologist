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

import glob
import os
from typing import Optional

try:
    import wandb  # type: ignore[import-untyped]
except ImportError:
    wandb = None  # type: ignore[assignment]


def pull_checkpoint(
    artifact: str,
    local_dir: str,
    storage_options: Optional[dict] = None,
) -> str:
    """Download a W&B artifact and return the local .ckpt path.

    Args:
        artifact: W&B artifact reference ``"entity/project/name:alias"``.
        local_dir: directory into which the artifact is downloaded.
        storage_options: unused; reserved for future fsspec pass-through.

    Returns:
        Absolute path to the downloaded ``.ckpt`` file.

    Raises:
        RuntimeError: if ``wandb`` is not installed.
        FileNotFoundError: if the downloaded artifact contains no ``.ckpt`` file.
    """
    if wandb is None:
        raise RuntimeError(
            "wandb is required to pull checkpoints. "
            "Install it with: pip install wandb"
        )

    api = wandb.Api()
    art = api.artifact(artifact)
    download_dir = art.download(root=local_dir)

    ckpt_files = glob.glob(os.path.join(download_dir, "**", "*.ckpt"), recursive=True)
    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt file found in artifact downloaded to {download_dir!r}"
        )

    return ckpt_files[0]
