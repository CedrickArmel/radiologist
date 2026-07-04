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

"""W&B model registry facade — resolve, download, push, and promote artifacts."""

from radiologist.registry.interface import ModelRegistry
from radiologist.registry.models import (
    ArtifactRef,
    CollectionMember,
    ExportResult,
    LoggedArtifacts,
    PromoteResult,
)
from radiologist.registry.selector import (
    RegistrySelector,
    resolve_selector,
    selector_from_flags,
)
from radiologist.registry.wandb_registry import WandbRegistry

__all__ = [
    "ArtifactRef",
    "CollectionMember",
    "ExportResult",
    "LoggedArtifacts",
    "ModelRegistry",
    "PromoteResult",
    "RegistrySelector",
    "WandbRegistry",
    "resolve_selector",
    "selector_from_flags",
]
