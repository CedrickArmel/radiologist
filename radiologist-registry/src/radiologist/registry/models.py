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

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ArtifactRef:
    """Resolved W&B artifact pointer — output of WandbRegistry.resolve()."""

    qualified_name: str
    run_id: str
    artifact_name: str
    version: str


@dataclass(frozen=True)
class ExportResult:
    """Paths to exported ONNX files — produced by core.export_onnx(), consumed by WandbRegistry.promote()."""

    det_path: str
    mcd_path: str
    run_id: str
    input_shape: Tuple[int, ...]
    classes: List[str]


@dataclass(frozen=True)
class PromoteResult:
    """Result of a link/transition transaction — det+mcd always share one alias."""

    det_qualified_name: str
    mcd_qualified_name: str
    alias: str


@dataclass(frozen=True)
class LoggedArtifacts:
    """Qualified names of the artifacts logged to the active run by the export callback.

    These are logged-but-not-linked: resolvable by run_id, carrying version aliases
    ('best'/'last'), but not yet attached to any registry collection.
    """

    det_qualified_name: str
    mcd_qualified_name: str
    run_id: str


@dataclass(frozen=True)
class CollectionMember:
    """One artifact version in a W&B collection together with its current alias list."""

    qualified_name: str
    aliases: List[str]
