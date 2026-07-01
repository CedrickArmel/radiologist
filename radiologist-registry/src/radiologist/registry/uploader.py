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
        raise NotImplementedError

    def link_to_collection(
        self,
        det_qualified_name: str,
        mcd_qualified_name: str,
        det_collection: str,
        mcd_collection: str,
        alias: str,
    ) -> PromoteResult:
        raise NotImplementedError
