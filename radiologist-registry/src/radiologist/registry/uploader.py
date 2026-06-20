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

from radiologist.registry.models import ExportResult, PromoteResult
from radiologist.registry.optional import _guard_wandb, _wandb


class _WandbUploader:
    """W&B seam for artifact upload operations."""

    def promote(
        self,
        export_result: ExportResult,
        collection: str,
        alias: str,
    ) -> PromoteResult:
        _guard_wandb()
        run = _wandb.init(job_type="registry-promote")  # type: ignore[union-attr]
        try:
            run_id = export_result.run_id

            det_art = _wandb.Artifact(f"model-{run_id}", type="model")  # type: ignore[union-attr]
            det_art.add_file(export_result.det_path)
            run.log_artifact(det_art)
            det_linked = run.link_artifact(det_art, collection, aliases=[alias])

            mcd_art = _wandb.Artifact(f"model-{run_id}-mcd", type="model")  # type: ignore[union-attr]
            mcd_art.add_file(export_result.mcd_path)
            run.log_artifact(mcd_art)
            mcd_linked = run.link_artifact(mcd_art, collection, aliases=[alias])

            return PromoteResult(
                det_qualified_name=det_linked.qualified_name,
                mcd_qualified_name=mcd_linked.qualified_name,
            )
        finally:
            run.finish()
