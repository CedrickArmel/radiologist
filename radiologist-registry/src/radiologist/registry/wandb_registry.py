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
from radiologist.registry.resolver import _WandbResolver
from radiologist.registry.uploader import _WandbUploader


class WandbRegistry:
    """Facade implementing the ModelRegistry Protocol by composing four W&B seams."""

    def __init__(self) -> None:
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
        return self._resolver.download(ref, local_dir)

    def pull(self, artifact_path: str, local_dir: str) -> str:
        return self._resolver.pull(artifact_path, local_dir)

    def log_model_artifacts(
        self,
        export_result: ExportResult,
        run: Any,
        ckpt_path: str,
        last_ckpt_path: Optional[str] = None,
    ) -> LoggedArtifacts:
        return self._uploader.log_model_artifacts(
            export_result, run, ckpt_path, last_ckpt_path
        )

    def list_collection_artifacts(
        self,
        type_name: str,
        collection_name: str,
    ) -> List[CollectionMember]:
        return self._lister.list_collection_artifacts(type_name, collection_name)

    def promote(
        self,
        path: str,
        run_id: str,
        det_collection: str,
        mcd_collection: str,
    ) -> PromoteResult:
        det_ref = self._resolver.resolve(path, run_id=run_id, version="best")
        mcd_ref = self._resolver.resolve(path, run_id=f"{run_id}-mcd", version="best")

        members = self._lister.list_collection_artifacts("model", det_collection)
        has_production = any("production" in m.aliases for m in members)
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
        det_members = self._lister.list_collection_artifacts("model", det_collection)
        mcd_members = self._lister.list_collection_artifacts("model", mcd_collection)

        det_staging = next((m for m in det_members if "staging" in m.aliases), None)
        mcd_staging = next((m for m in mcd_members if "staging" in m.aliases), None)
        if det_staging is None or mcd_staging is None:
            raise LookupError(
                "No 'staging' member found in one or both collections: "
                f"{det_collection!r}, {mcd_collection!r}"
            )

        det_production = next(
            (m for m in det_members if "production" in m.aliases), None
        )
        mcd_production = next(
            (m for m in mcd_members if "production" in m.aliases), None
        )

        if det_production is not None:
            self._alias_manager.remove_alias(
                det_production.qualified_name, "production"
            )
        self._alias_manager.set_alias(det_staging.qualified_name, "production")

        if mcd_production is not None:
            self._alias_manager.remove_alias(
                mcd_production.qualified_name, "production"
            )
        self._alias_manager.set_alias(mcd_staging.qualified_name, "production")

        return PromoteResult(
            det_qualified_name=det_staging.qualified_name,
            mcd_qualified_name=mcd_staging.qualified_name,
            alias="production",
        )

    def get_aliases(self, artifact_path: str) -> List[str]:
        return self._alias_manager.get_aliases(artifact_path)

    def set_alias(self, artifact_path: str, alias: str) -> None:
        self._alias_manager.set_alias(artifact_path, alias)

    def remove_alias(self, artifact_path: str, alias: str) -> None:
        self._alias_manager.remove_alias(artifact_path, alias)
