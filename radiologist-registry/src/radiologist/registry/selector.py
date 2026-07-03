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
        return bool(
            self.run_id or self.tags or self.groups or self.metric or self.version
        )


def selector_from_flags(
    path: str,
    run_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    metric: Optional[str] = None,
    version: Optional[str] = None,
    include_sweeps: bool = False,
) -> RegistrySelector:
    """Build a RegistrySelector from CLI-style flag values."""
    return RegistrySelector(
        path=path,
        run_id=run_id,
        tags=tags,
        groups=groups,
        metric=metric,
        version=version,
        include_sweeps=include_sweeps,
    )


def resolve_selector(
    selector: RegistrySelector, registry: ModelRegistry
) -> ArtifactRef:
    if selector.run_id and selector.tags:
        raise ValueError("Provide either --run-id or --tags, not both.")
    return registry.resolve(
        path=selector.path,
        run_id=selector.run_id,
        groups=selector.groups,
        tags=selector.tags,
        metric=selector.metric,
        version=selector.version,
        include_sweeps=selector.include_sweeps,
    )
