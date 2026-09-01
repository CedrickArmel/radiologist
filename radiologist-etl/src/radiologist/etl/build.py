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

"""Build stage: one split manifest → WebDataset tar shards + manifest + report.

Writes shards, a shard-annotated manifest, and a split report under
``{shard_root}/{run_id}``. Never mutates the input split manifest. The
``ratios`` argument is used only for the report's configured-vs-observed
comparison — never for assignment (assignment already happened in the
assign-split stage).

Note: this module's :func:`build_shards` is the new pure-function public API
for the build stage (issue #185 implements its body); it is deliberately
*not* re-exported from ``radiologist.etl`` yet, since the package still
exports the existing, real, working
:func:`radiologist.etl.shards.build_shards` (a different signature) that the
still-live monolithic ``etl_flow`` and the current test suite depend on. The
package-level rebind happens once #185 lands the real implementation here.
"""

from __future__ import annotations

from radiologist.etl.execution import ShardMapper
from radiologist.etl.models import BuildResult
from radiologist.etl.split import SplitRatios


def build_shards(
    split_manifest_path: str,
    shard_root: str,
    shard_size: int = 1000,
    ratios: SplitRatios | None = None,
    workers: int | None = None,
    run_label: str | None = None,
    mapper: ShardMapper | None = None,
    storage_options: dict | None = None,
) -> BuildResult:
    """Build WebDataset tar shards, a shard-annotated manifest, and a split report.

    Args:
        split_manifest_path: path to the split manifest produced by the
            assign-split stage.
        shard_root: directory shards are written under (inside
            ``{shard_root}/{run_id}``).
        shard_size: max samples per shard.
        ratios: configured split ratios, used only for the report.
        workers: worker count; defaults to :func:`~radiologist.etl.execution.default_workers`.
        run_label: optional label folded into the run id.
        mapper: shard-dispatching callable; defaults to a local process-pool mapper.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A :class:`~radiologist.etl.models.BuildResult` describing the run.
    """
    raise NotImplementedError
