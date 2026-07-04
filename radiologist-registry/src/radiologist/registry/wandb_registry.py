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

"""W&B-backed implementation of the ModelRegistry Protocol."""

from typing import Any, List, Optional, Union

from radiologist.registry.alias_manager import _WandbAliasManager
from radiologist.registry.collection import _WandbCollectionLister
from radiologist.registry.models import (
    ArtifactRef,
    CollectionMember,
    ExportResult,
    LoggedArtifacts,
    PromoteResult,
)
from radiologist.registry.optional import _MODEL_ARTIFACT_TYPE
from radiologist.registry.resolver import _WandbResolver
from radiologist.registry.uploader import _WandbUploader


def _find_by_alias(
    members: List[CollectionMember], alias: str
) -> Optional[CollectionMember]:
    return next((m for m in members if alias in m.aliases), None)


class WandbRegistry:
    """Facade implementing the ModelRegistry Protocol by composing four W&B seams."""

    def __init__(self) -> None:
        """Construct eagerly — no injected dependencies needed for CLI use."""
        self._resolver = _WandbResolver()
        self._uploader = _WandbUploader()
        self._alias_manager = _WandbAliasManager()
        self._lister = _WandbCollectionLister()

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
        return self._resolver.resolve(
            path=path,
            run_id=run_id,
            groups=groups,
            tags=tags,
            metric=metric,
            version=version,
            include_sweeps=include_sweeps,
        )

    def download(self, ref: ArtifactRef, local_dir: str) -> str:
        """Download the checkpoint file (``*.ckpt``) for an artifact.

        Args:
            ref: Previously resolved artifact pointer.
            local_dir: Directory to download the artifact contents into.

        Returns:
            Filesystem path to the downloaded ``.ckpt`` file.
        """
        return self._resolver.download(ref, local_dir)

    def pull(self, artifact_path: str, local_dir: str) -> str:
        """Download the ONNX file (``*.onnx``) for a qualified artifact path.

        Args:
            artifact_path: Fully qualified artifact path.
            local_dir: Directory to download the artifact contents into.

        Returns:
            Filesystem path to the downloaded ``.onnx`` file.
        """
        return self._resolver.pull(artifact_path, local_dir)

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
        return self._uploader.log_model_artifacts(
            export_result, run, ckpt_path, last_ckpt_path
        )

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
        return self._lister.list_collection_artifacts(type_name, collection_name)

    def promote(
        self,
        path: str,
        run_id: str,
        det_collection: str,
        mcd_collection: str,
    ) -> PromoteResult:
        """Link a run's deterministic and MC-Dropout artifacts to collections.

        The shared alias is ``"production"`` unless either collection already
        has a member aliased ``"production"``, in which case it is
        ``"staging"`` — new runs never silently overwrite a live production
        model.

        Args:
            path: Base artifact path shared by both artifacts.
            run_id: Run whose ``"best"`` artifacts should be promoted. The
                MC-Dropout artifact is looked up under ``f"{run_id}-mcd"``.
            det_collection: Collection to link the deterministic artifact to.
            mcd_collection: Collection to link the MC-Dropout artifact to.

        Returns:
            The alias applied and the qualified names of both linked
            artifacts.
        """
        det_ref = self._resolver.resolve(path, run_id=run_id, version="best")
        mcd_ref = self._resolver.resolve(path, run_id=f"{run_id}-mcd", version="best")

        det_members = self._lister.list_collection_artifacts(
            _MODEL_ARTIFACT_TYPE, det_collection
        )
        mcd_members = self._lister.list_collection_artifacts(
            _MODEL_ARTIFACT_TYPE, mcd_collection
        )
        has_production = any(
            "production" in m.aliases for m in (*det_members, *mcd_members)
        )
        alias = "staging" if has_production else "production"

        return self._uploader.link_to_collection(
            det_ref.qualified_name,
            mcd_ref.qualified_name,
            det_collection,
            mcd_collection,
            alias,
        )

    def transition_to_production(
        self,
        det_collection: str,
        mcd_collection: str,
    ) -> PromoteResult:
        """Promote the ``"staging"`` member of each collection to ``"production"``.

        Any existing ``"production"`` member has that alias removed first.
        If applying the new alias to the MC-Dropout artifact fails, the
        deterministic artifact's alias change is rolled back before the
        error is re-raised, so the two collections don't end up out of
        sync.

        Args:
            det_collection: Collection holding the deterministic artifact.
            mcd_collection: Collection holding the MC-Dropout artifact.

        Returns:
            The artifacts now carrying the ``"production"`` alias.

        Raises:
            LookupError: If either collection has no ``"staging"`` member.
        """
        det_members = self._lister.list_collection_artifacts(
            _MODEL_ARTIFACT_TYPE, det_collection
        )
        mcd_members = self._lister.list_collection_artifacts(
            _MODEL_ARTIFACT_TYPE, mcd_collection
        )

        det_staging = _find_by_alias(det_members, "staging")
        mcd_staging = _find_by_alias(mcd_members, "staging")
        if det_staging is None or mcd_staging is None:
            raise LookupError(
                "No 'staging' member found in one or both collections: "
                f"{det_collection!r}, {mcd_collection!r}"
            )

        det_production = _find_by_alias(det_members, "production")
        mcd_production = _find_by_alias(mcd_members, "production")

        if det_production is not None:
            self._alias_manager.remove_alias(
                det_production.qualified_name, "production"
            )
        self._alias_manager.set_alias(det_staging.qualified_name, "production")

        try:
            if mcd_production is not None:
                self._alias_manager.remove_alias(
                    mcd_production.qualified_name, "production"
                )
            self._alias_manager.set_alias(mcd_staging.qualified_name, "production")
        except Exception:
            try:
                self._alias_manager.remove_alias(
                    det_staging.qualified_name, "production"
                )
                if det_production is not None:
                    self._alias_manager.set_alias(
                        det_production.qualified_name, "production"
                    )
            except Exception:
                pass
            raise

        return PromoteResult(
            det_qualified_name=det_staging.qualified_name,
            mcd_qualified_name=mcd_staging.qualified_name,
            alias="production",
        )

    def get_aliases(self, artifact_path: str) -> List[str]:
        """Return the current alias list of an artifact.

        Args:
            artifact_path: Fully qualified artifact path.

        Returns:
            The artifact's current aliases.
        """
        return self._alias_manager.get_aliases(artifact_path)

    def set_alias(self, artifact_path: str, alias: str) -> None:
        """Add an alias to an artifact, if not already present.

        Args:
            artifact_path: Fully qualified artifact path.
            alias: Alias to add.
        """
        self._alias_manager.set_alias(artifact_path, alias)

    def remove_alias(self, artifact_path: str, alias: str) -> None:
        """Remove an alias from an artifact, if present.

        Args:
            artifact_path: Fully qualified artifact path.
            alias: Alias to remove.
        """
        self._alias_manager.remove_alias(artifact_path, alias)
