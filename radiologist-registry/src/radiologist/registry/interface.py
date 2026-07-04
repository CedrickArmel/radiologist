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

"""Structural contract that any registry backend must satisfy."""

from typing import Any, List, Optional, Protocol, Union

from radiologist.registry.models import (
    ArtifactRef,
    CollectionMember,
    ExportResult,
    LoggedArtifacts,
    PromoteResult,
)


class ModelRegistry(Protocol):
    """Contract for a model registry backend."""

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
        """Resolve a single artifact matching the given criteria.

        Args:
            path: Base artifact path, e.g. ``"entity/project"`` or
                ``"entity/project/artifact-name"``.
            run_id: If given, resolve the artifact logged by this run
                directly instead of searching.
            groups: Restrict the run search to these W&B group name(s).
            tags: Restrict the run search to runs carrying these tag(s).
            metric: Summary metric name used to rank candidate runs
                (highest first) when searching by tags.
            version: Explicit version or alias to resolve (defaults to
                ``"best"`` when omitted).
            include_sweeps: Whether runs that are part of a sweep are
                eligible candidates.

        Returns:
            The resolved artifact pointer.
        """
        ...

    def download(self, ref: ArtifactRef, local_dir: str) -> str:
        """Download the checkpoint file (``*.ckpt``) for an artifact.

        Args:
            ref: Previously resolved artifact pointer.
            local_dir: Directory to download the artifact contents into.

        Returns:
            Filesystem path to the downloaded ``.ckpt`` file.
        """
        ...

    def pull(self, artifact_path: str, local_dir: str) -> str:
        """Download the ONNX file (``*.onnx``) for a qualified artifact path.

        Args:
            artifact_path: Fully qualified artifact path.
            local_dir: Directory to download the artifact contents into.

        Returns:
            Filesystem path to the downloaded ``.onnx`` file.
        """
        ...

    def log_model_artifacts(
        self,
        export_result: ExportResult,
        run: Any,
        ckpt_path: str,
        last_ckpt_path: Optional[str] = None,
    ) -> LoggedArtifacts:
        """Log the deterministic and MC-Dropout exports to an active run.

        Args:
            export_result: Paths and metadata for the freshly exported
                model pair.
            run: Active run object used to log the artifacts.
            ckpt_path: Checkpoint path bundled with the deterministic
                artifact.
            last_ckpt_path: Optional path to the last (non-best) checkpoint,
                logged under the ``"last"`` alias when given.

        Returns:
            Qualified names of the artifacts just logged.
        """
        ...

    def list_collection_artifacts(
        self,
        type_name: str,
        collection_name: str,
    ) -> List[CollectionMember]:
        """List every artifact version in a collection with its aliases.

        Args:
            type_name: Artifact type of the collection (e.g. ``"model"``).
            collection_name: Name of the collection to list.

        Returns:
            One `CollectionMember` per artifact version in the collection.
        """
        ...

    def promote(
        self,
        path: str,
        run_id: str,
        det_collection: str,
        mcd_collection: str,
    ) -> PromoteResult:
        """Link a run's deterministic and MC-Dropout artifacts to collections.

        Args:
            path: Base artifact path shared by both artifacts.
            run_id: Run whose ``"best"`` artifacts should be promoted.
            det_collection: Collection to link the deterministic artifact to.
            mcd_collection: Collection to link the MC-Dropout artifact to.

        Returns:
            The alias (``"staging"`` or ``"production"``) applied and the
            qualified names of both linked artifacts.
        """
        ...

    def transition_to_production(
        self,
        det_collection: str,
        mcd_collection: str,
    ) -> PromoteResult:
        """Promote the ``"staging"`` member of each collection to ``"production"``.

        Args:
            det_collection: Collection holding the deterministic artifact.
            mcd_collection: Collection holding the MC-Dropout artifact.

        Returns:
            The artifacts now carrying the ``"production"`` alias.

        Raises:
            LookupError: If either collection has no ``"staging"`` member.
        """
        ...

    def get_aliases(self, artifact_path: str) -> List[str]:
        """Return the current alias list of an artifact.

        Args:
            artifact_path: Fully qualified artifact path.

        Returns:
            The artifact's current aliases.
        """
        ...

    def set_alias(self, artifact_path: str, alias: str) -> None:
        """Add an alias to an artifact, if not already present.

        Args:
            artifact_path: Fully qualified artifact path.
            alias: Alias to add.
        """
        ...

    def remove_alias(self, artifact_path: str, alias: str) -> None:
        """Remove an alias from an artifact, if present.

        Args:
            artifact_path: Fully qualified artifact path.
            alias: Alias to remove.
        """
        ...
