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

"""Framework-neutral artifact selector shared by the CLI and library layers."""

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

    Attributes:
        path: Base artifact path, e.g. ``"entity/project"`` or a raw
            artifact path when not registry-backed.
        run_id: If given, resolve the artifact logged by this run directly.
        groups: Restrict the run search to these W&B group name(s).
        tags: Restrict the run search to runs carrying these tag(s).
        metric: Summary metric name used to rank candidate runs.
        version: Explicit version or alias to resolve.
        include_sweeps: Whether sweep runs are eligible candidates.
    """

    path: str
    run_id: Optional[str] = None
    groups: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    metric: Optional[str] = None
    version: Optional[str] = None
    include_sweeps: bool = False

    def is_registry_backed(self) -> bool:
        """Report whether this selector carries any registry search criteria.

        Returns:
            True if any of `run_id`, `tags`, `groups`, `metric`, or
            `version` is set — meaning the selector should be resolved via
            the registry rather than treated as a raw path.
        """
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
    """Build a RegistrySelector from CLI-style flag values.

    Args:
        path: Base artifact path.
        run_id: Direct run ID to resolve, if any.
        tags: Tag(s) to filter candidate runs by.
        groups: Group(s) to filter candidate runs by.
        metric: Summary metric used to rank candidate runs.
        version: Explicit version or alias to resolve.
        include_sweeps: Whether sweep runs are eligible candidates.

    Returns:
        The constructed selector.
    """
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
    """Validate a selector and resolve it against a registry backend.

    Args:
        selector: Selector describing which artifact to resolve.
        registry: Registry backend used to perform the resolution.

    Returns:
        The resolved artifact pointer.

    Raises:
        ValueError: If both `run_id` and `tags` are set on the selector.
    """
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
