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
from typing import List, Optional

from radiologist.registry.interface import ModelRegistry
from radiologist.registry.models import ArtifactRef


@dataclass(frozen=True)
class RegistrySelector:
    """Declarative description of which artifact to resolve, framework-neutral.

    Precedence when resolved: run_id (direct lookup) else tags (filtered search)
    else path (raw artifact path / local file). See resolve_selector for the
    strict validation the CLI layer adds on top of the underlying resolve().
    """

    path: str
    run_id: Optional[str] = None
    groups: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    metric: Optional[str] = None
    version: Optional[str] = None
    include_sweeps: bool = False

    def is_registry_backed(self) -> bool:
        # contract: True when any registry selector field (run_id/tags/groups/
        #   metric/version) is set; False when only a raw path/local file is
        #   implied. Single dispatch rule for "local file vs registry".
        raise NotImplementedError


def resolve_selector(
    selector: RegistrySelector, registry: ModelRegistry
) -> ArtifactRef:
    # contract: forwards selector fields to registry.resolve(...) applying the
    #   run_id -> tags -> path cascade and returns the resolved ArtifactRef.
    #   Raises ValueError when BOTH run_id and tags are set (strict CLI-facing
    #   validation, stricter than the underlying resolve()). Propagates whatever
    #   registry.resolve raises (ValueError on no match, RuntimeError when wandb
    #   missing).
    raise NotImplementedError
