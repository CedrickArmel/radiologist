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

"""Pure readers over this uv workspace's own package metadata.

Answers three questions with no network access and no third-party parser:
which distributions the workspace declares, which intra-workspace
requirements are unpinned or carry a stale floor, and what a real-PyPI
resolver would see for a given distribution. Consumed by
``scripts/release_bump.py`` (roster), by the executable contracts in
``scripts_tests/``, and by ``publish.yml``'s resolution guard.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - this repo pins Python 3.10
    import tomli as tomllib  # type: ignore[import-untyped,no-redef] # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PACKAGE = "radiologist"


@dataclass(frozen=True)
class IntraWorkspaceRequirement:
    """One declared dependency edge from a workspace member onto another."""

    consumer: str  # distribution declaring the dependency
    target: str  # workspace distribution depended upon
    extras: Tuple[str, ...]  # extras requested on the target, e.g. ("all",)
    specifier: str  # ">=0.1.0"; "" when no specifier is declared
    origin: str  # "dependencies" | "optional-dependencies.<extra>"
    raw: str  # the requirement string exactly as written


def _root_manifest(repo_root: Path) -> Dict:
    with (Path(repo_root) / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def workspace_packages(repo_root: Path) -> Tuple[str, ...]:
    """Return every distribution name declared by this uv workspace.

    Returns the root distribution name followed by every entry of
    ``[tool.uv.workspace] members``, in declaration order. Names are
    returned exactly as declared. Raises ``KeyError`` if the root manifest
    declares no ``[tool.uv.workspace] members`` key.
    """
    data = _root_manifest(repo_root)
    members = data["tool"]["uv"]["workspace"]["members"]
    return (ROOT_PACKAGE, *members)


def package_dir(repo_root: Path, package: str) -> str:
    """Return the on-disk directory for ``package`` relative to ``repo_root``.

    ``"."`` for the root distribution, otherwise the member's directory name
    (which equals its distribution name). Raises ``ValueError`` for a name
    not in :func:`workspace_packages`.
    """
    packages = workspace_packages(repo_root)
    if package not in packages:
        raise ValueError(
            f"Unknown distribution {package!r}; expected one of {packages}"
        )
    return "." if package == ROOT_PACKAGE else package


def package_manifest_path(repo_root: Path, package: str) -> Path:
    """Return the path to ``package``'s ``pyproject.toml``.

    ``repo_root/pyproject.toml`` for the root distribution, otherwise
    ``repo_root/<package>/pyproject.toml``. Raises ``ValueError`` for a name
    not in :func:`workspace_packages`.
    """
    return Path(repo_root) / package_dir(repo_root, package) / "pyproject.toml"


def _load_manifest(repo_root: Path, package: str) -> Dict:
    manifest_path = package_manifest_path(repo_root, package)
    with manifest_path.open("rb") as handle:
        return tomllib.load(handle)


def declared_version(repo_root: Path, package: str) -> str:
    """Return ``[project].version`` from ``package``'s manifest."""
    return str(_load_manifest(repo_root, package)["project"]["version"])


_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*(.*)$")
_FLOOR_RE = re.compile(r">=\s*([0-9][0-9A-Za-z.]*)")


def _normalize(name: str) -> str:
    """Normalise a distribution name per PEP 503 for comparison."""
    return name.lower().replace("_", "-")


def _parse_requirement(raw: str) -> Tuple[str, Tuple[str, ...], str]:
    """Split a PEP 508 requirement string into name, extras and specifier."""
    match = _REQUIREMENT_RE.match(raw.strip())
    if not match:
        raise ValueError(f"Cannot parse requirement: {raw!r}")
    name, extras_group, specifier = match.groups()
    extras = (
        tuple(e.strip() for e in extras_group[1:-1].split(",") if e.strip())
        if extras_group
        else ()
    )
    return name, extras, specifier.strip()


def _iter_manifest_requirement_entries(manifest: Dict) -> Iterable[Tuple[str, str]]:
    """Yield ``(origin, raw_requirement)`` for every declared requirement."""
    project = manifest.get("project", {})
    for raw in project.get("dependencies", []):
        yield "dependencies", raw
    optional_dependencies = project.get("optional-dependencies", {})
    for extra, requirements in optional_dependencies.items():
        for raw in requirements:
            yield f"optional-dependencies.{extra}", raw


def intra_workspace_requirements(
    repo_root: Path, package: str
) -> List[IntraWorkspaceRequirement]:
    """Return every requirement of ``package`` naming another workspace member.

    Covers default dependencies and each extra. Requirements naming
    third-party distributions are omitted. Order is stable: default
    dependencies first, then extras in declaration order, each in
    declaration order.
    """
    manifest = _load_manifest(repo_root, package)
    normalized_members = {_normalize(name) for name in workspace_packages(repo_root)}
    result = []
    for origin, raw in _iter_manifest_requirement_entries(manifest):
        name, extras, specifier = _parse_requirement(raw)
        if _normalize(name) in normalized_members:
            result.append(
                IntraWorkspaceRequirement(
                    consumer=package,
                    target=name,
                    extras=extras,
                    specifier=specifier,
                    origin=origin,
                    raw=raw,
                )
            )
    return result


def unpinned_requirements(repo_root: Path) -> List[IntraWorkspaceRequirement]:
    """Return every intra-workspace requirement across the workspace with no floor.

    Empty list when every edge is pinned.
    """
    result = []
    for package in workspace_packages(repo_root):
        for requirement in intra_workspace_requirements(repo_root, package):
            if not requirement.specifier:
                result.append(requirement)
    return result


def _floor_version(specifier: str) -> Optional[str]:
    match = _FLOOR_RE.match(specifier)
    return match.group(1) if match else None


def _version_tuple(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def stale_pin_floors(repo_root: Path) -> List[IntraWorkspaceRequirement]:
    """Return every intra-workspace requirement whose lower bound is stale.

    Advisory input only -- a stale floor is legal packaging, not an error.
    Requirements with no lower bound at all are not reported here
    (:func:`unpinned_requirements` owns them).
    """
    result = []
    for package in workspace_packages(repo_root):
        for requirement in intra_workspace_requirements(repo_root, package):
            floor = _floor_version(requirement.specifier)
            if floor is None:
                continue
            target_version = declared_version(repo_root, requirement.target)
            if _version_tuple(floor) < _version_tuple(target_version):
                result.append(requirement)
    return result


def publishable_requirement_lines(repo_root: Path, package: str) -> List[str]:
    """Return the PEP 508 requirement lines a real-PyPI resolver would see.

    Covers ``package``'s default dependencies plus every extra's,
    deduplicated and sorted. ``[tool.uv.sources]`` is NOT applied. Returns
    ``[]`` when the distribution declares no dependency at all.
    """
    manifest = _load_manifest(repo_root, package)
    lines = {raw for _, raw in _iter_manifest_requirement_entries(manifest)}
    return sorted(lines)


def render_stale_pin_markdown(
    findings: List[IntraWorkspaceRequirement],
    target_versions: Dict[str, str],
) -> str:
    """Render ``findings`` as a GitHub-flavoured markdown table.

    Prefixed with the stable HTML marker comment
    ``<!-- radiologist:pin-cascade-advisory -->`` so a workflow can update
    its own prior comment in place. Returns a short "nothing stale" body
    (still marker-prefixed) for an empty list.
    """
    raise NotImplementedError


def _main(argv: List[str]) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
