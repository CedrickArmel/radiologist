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

"""Internal W&B seam for listing registry collection members."""

from typing import List

from radiologist.registry.models import CollectionMember
from radiologist.registry.optional import _guard_wandb, _wandb  # noqa: F401


class _WandbCollectionLister:
    """W&B seam for listing the members (and aliases) of a registry collection."""

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
        _guard_wandb()
        api = _wandb.Api()  # type: ignore[union-attr]
        collection = api.artifact_collection(type_name, collection_name)
        return [
            CollectionMember(
                qualified_name=art.qualified_name, aliases=list(art.aliases)
            )
            for art in collection.artifacts()
        ]
