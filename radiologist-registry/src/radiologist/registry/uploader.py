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

from typing import Any, Optional

from radiologist.registry.models import ExportResult, LoggedArtifacts, PromoteResult
from radiologist.registry.optional import _guard_wandb, _wandb  # noqa: F401


class _WandbUploader:
    """W&B seam for artifact upload operations."""

    def log_model_artifacts(
        self,
        export_result: ExportResult,
        run: Any,
        ckpt_path: str,
        last_ckpt_path: Optional[str] = None,
    ) -> LoggedArtifacts:
        _guard_wandb()
        run_id = export_result.run_id
        det_name = f"model-{run_id}"
        mcd_name = f"model-{run_id}-mcd"

        det_art = _wandb.Artifact(det_name, type="model")  # type: ignore[union-attr]
        det_art.add_file(export_result.det_path)
        det_art.add_file(ckpt_path)
        run.log_artifact(det_art, aliases=["best"])

        mcd_art = _wandb.Artifact(mcd_name, type="model")  # type: ignore[union-attr]
        mcd_art.add_file(export_result.mcd_path)
        run.log_artifact(mcd_art, aliases=["best"])

        if last_ckpt_path:
            last_art = _wandb.Artifact(det_name, type="model")  # type: ignore[union-attr]
            last_art.add_file(last_ckpt_path)
            run.log_artifact(last_art, aliases=["last"])

        entity = getattr(run, "entity", "")
        project = getattr(run, "project", "")
        return LoggedArtifacts(
            det_qualified_name=f"{entity}/{project}/{det_name}:best",
            mcd_qualified_name=f"{entity}/{project}/{mcd_name}:best",
            run_id=run_id,
        )

    def link_to_collection(
        self,
        det_qualified_name: str,
        mcd_qualified_name: str,
        det_collection: str,
        mcd_collection: str,
        alias: str,
    ) -> PromoteResult:
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]

        det_art = api.artifact(det_qualified_name)
        det_art.link(det_collection, aliases=[alias])

        mcd_art = api.artifact(mcd_qualified_name)
        mcd_art.link(mcd_collection, aliases=[alias])

        return PromoteResult(
            det_qualified_name=det_qualified_name,
            mcd_qualified_name=mcd_qualified_name,
            alias=alias,
        )
