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

from __future__ import annotations

import glob
import os
from typing import List, Optional, Union

try:
    import wandb  # type: ignore[import-untyped]
except ImportError:
    wandb = None  # type: ignore[assignment]


def _resolve_model_artifact(
    path: str,
    run_id: Optional[str] = None,
    groups: Optional[Union[str, List[str]]] = None,
    tags: Optional[Union[str, List[str]]] = None,
    metric: Optional[str] = None,
    version: Optional[str] = None,
    include_sweeps: bool = False,
) -> "wandb.Artifact":
    api = wandb.Api()  # type: ignore[union-attr]
    if run_id:
        artifact = api.artifact(
            type="model", name=f"{path}/model-{run_id}:{version or 'best'}"
        )
    elif tags:
        if isinstance(tags, str):
            tags = [tags]
        filters: dict = {"tags": {"$in": tags}}
        if groups:
            if isinstance(groups, str):
                groups = [groups]
            filters["group"] = {"$in": groups}
        runs = api.runs(
            path=path,
            filters=filters,
            order=f"-summary_metric.{metric or 'best_val_score'}",
            include_sweeps=include_sweeps,
        )
        best_run = max(
            runs, key=lambda run: run.summary.get(metric or "best_val_score", 0)
        )
        artifact = api.artifact(
            type="model", name=f"{path}/model-{best_run.id}:{version or 'best'}"
        )
    else:
        artifact = api.artifact(path)

    return artifact


def pull_checkpoint(
    path: str,
    local_dir: str,
    run_id: Optional[str] = None,
    groups: Optional[Union[str, List[str]]] = None,
    tags: Optional[Union[str, List[str]]] = None,
    metric: Optional[str] = None,
    version: Optional[str] = None,
    include_sweeps: bool = False,
) -> str:
    if wandb is None:
        raise RuntimeError("wandb is not installed; run `pip install wandb`")
    art = _resolve_model_artifact(
        path=path,
        run_id=run_id,
        groups=groups,
        tags=tags,
        metric=metric,
        version=version,
        include_sweeps=include_sweeps,
    )
    download_dir = art.download(root=local_dir)
    ckpt_files = glob.glob(os.path.join(download_dir, "**", "*.ckpt"), recursive=True)
    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt file found in artifact downloaded to {download_dir!r}"
        )
    return ckpt_files[0]
