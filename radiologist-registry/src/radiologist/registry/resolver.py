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

import glob
import os
from typing import List, Optional, Union

from radiologist.registry.models import ArtifactRef
from radiologist.registry.optional import _guard_wandb, _wandb


def _artifact_ref(art: object, run_id: str) -> ArtifactRef:
    qualified_name: str = art.qualified_name  # type: ignore[attr-defined]
    version: str = art.version  # type: ignore[attr-defined]
    artifact_name = qualified_name.split("/")[-1].split(":")[0]
    return ArtifactRef(
        qualified_name=qualified_name,
        run_id=run_id,
        artifact_name=artifact_name,
        version=version,
    )


class _WandbResolver:
    """W&B seam for artifact resolution and download operations."""

    def resolve(
        self,
        path: str,
        run_id: Optional[str] = None,
        groups: Optional[Union[str, List[str]]] = None,
        tags: Optional[Union[str, List[str]]] = None,
        metric: Optional[str] = None,
        version: Optional[str] = None,
        include_sweeps: bool = False,
    ) -> ArtifactRef:
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]

        if run_id:
            art = api.artifact(
                type="model",
                name=f"{path}/model-{run_id}:{version or 'best'}",
            )
            return _artifact_ref(art, run_id)

        if tags:
            if isinstance(tags, str):
                tags = [tags]
            filters: dict = {"tags": {"$in": tags}}
            if groups:
                if isinstance(groups, str):
                    groups = [groups]
                filters["group"] = {"$in": groups}
            kwargs: dict = {
                "path": path,
                "filters": filters,
                "include_sweeps": include_sweeps,
            }
            if metric is not None:
                kwargs["order"] = f"-summary_metric.{metric}"
            runs = api.runs(**kwargs)
            best_run = next(iter(runs))
            art = api.artifact(
                type="model",
                name=f"{path}/model-{best_run.id}:{version or 'best'}",
            )
            return _artifact_ref(art, best_run.id)

        art = api.artifact(path)
        run = art.logged_by()
        run_id = run.id if run is not None else ""
        return _artifact_ref(art, run_id)

    def download(self, ref: ArtifactRef, local_dir: str) -> str:
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]
        art = api.artifact(ref.qualified_name)
        download_dir = art.download(root=local_dir)
        ckpt_files = glob.glob(
            os.path.join(download_dir, "**", "*.ckpt"), recursive=True
        )
        if not ckpt_files:
            raise FileNotFoundError(
                f"No .ckpt file found in artifact downloaded to {download_dir!r}"
            )
        return ckpt_files[0]

    def pull(self, artifact_path: str, local_dir: str) -> str:
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]
        art = api.artifact(artifact_path)
        download_dir = art.download(local_dir)
        onnx_files = [
            os.path.join(download_dir, f)
            for f in os.listdir(download_dir)
            if f.endswith(".onnx")
        ]
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx file found in artifact downloaded to {download_dir!r}"
            )
        return onnx_files[0]
