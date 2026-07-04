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

"""Frozen dataclasses exchanged across the registry facade's public API.

These carry no behavior — they are plain, immutable value objects passed
between `WandbRegistry`, `resolve_selector`, and the CLI layer.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ArtifactRef:
    """Resolved W&B artifact pointer — output of WandbRegistry.resolve().

    Attributes:
        qualified_name: Fully qualified artifact path, e.g.
            ``"entity/project/model-run123:best"``.
        run_id: W&B run ID that produced the artifact.
        artifact_name: Bare artifact name without entity/project/version,
            e.g. ``"model-run123"``.
        version: Resolved version or alias, e.g. ``"best"`` or ``"v3"``.
    """

    qualified_name: str
    run_id: str
    artifact_name: str
    version: str


@dataclass(frozen=True)
class ExportResult:
    """Paths to exported ONNX files — produced by core.export_onnx().

    Consumed by WandbRegistry.log_model_artifacts().

    Attributes:
        det_path: Filesystem path to the deterministic ONNX export.
        mcd_path: Filesystem path to the MC-Dropout ONNX export.
        run_id: W&B run ID the exports belong to.
        input_shape: Model input tensor shape, e.g. ``(1, 3, 224, 224)``.
        classes: Ordered list of class labels the model predicts.
    """

    det_path: str
    mcd_path: str
    run_id: str
    input_shape: Tuple[int, ...]
    classes: List[str]


@dataclass(frozen=True)
class PromoteResult:
    """Result of a link/transition transaction — det+mcd always share one alias.

    Attributes:
        det_qualified_name: Qualified name of the deterministic artifact
            after the transaction.
        mcd_qualified_name: Qualified name of the MC-Dropout artifact after
            the transaction.
        alias: Alias applied to both artifacts (``"staging"`` or
            ``"production"``).
    """

    det_qualified_name: str
    mcd_qualified_name: str
    alias: str


@dataclass(frozen=True)
class LoggedArtifacts:
    """Qualified names of the artifacts logged to the active run by the export callback.

    These are logged-but-not-linked: resolvable by run_id, carrying version aliases
    ('best'/'last'), but not yet attached to any registry collection.

    Attributes:
        det_qualified_name: Qualified name of the freshly logged deterministic
            artifact.
        mcd_qualified_name: Qualified name of the freshly logged MC-Dropout
            artifact.
        run_id: W&B run ID the artifacts were logged under.
    """

    det_qualified_name: str
    mcd_qualified_name: str
    run_id: str


@dataclass(frozen=True)
class CollectionMember:
    """One artifact version in a W&B collection together with its current alias list.

    Attributes:
        qualified_name: Fully qualified artifact path of this member.
        aliases: Aliases currently attached to this artifact version.
    """

    qualified_name: str
    aliases: List[str]
