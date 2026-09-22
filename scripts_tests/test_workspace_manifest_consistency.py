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

"""Executable sync-check contract for the workspace's release/publish surfaces.

Issue #227 already keeps ``release_bump.PACKAGES`` derived from the workspace
manifest, and ``test_release_publish_package_options.py`` already pins down
that ``radiologist-cli`` is present in both ``release.yml`` and
``publish.yml``'s ``package.options`` dropdowns.

This module is issue #229's own contract: it fails the suite -- naming the
offending distribution and, for the workflow-dropdown check, which workflow
is missing it -- the moment any of the three release/publish surfaces (the
workspace manifest, ``release_bump.PACKAGES``, and the two workflow
dropdowns) drift apart from each other, and the moment any intra-workspace
requirement loses its version floor.
"""

import re
from pathlib import Path
from typing import Set

import pytest
import release_bump
import workspace_manifests

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

_OPTIONS_RE = re.compile(
    r"package:\n(?:.*\n)*?\s*options:\n((?:\s*-\s*\S+\n)+)", re.MULTILINE
)


def _package_choice_options(workflow_path: Path) -> Set[str]:
    text = workflow_path.read_text()
    match = _OPTIONS_RE.search(text)
    assert match, f"{workflow_path.name}: could not find package.options block"
    return {
        line.strip("- ").strip() for line in match.group(1).splitlines() if line.strip()
    }


def test_every_workspace_distribution_is_releasable() -> None:
    """The release script's roster is exactly the workspace's own roster."""
    declared = set(workspace_manifests.workspace_packages(_REPO_ROOT))
    released = set(release_bump.PACKAGES)
    assert released == declared, (
        "release_bump.PACKAGES is out of sync with the workspace manifest; "
        f"missing: {declared - released}, extra: {released - declared}"
    )


@pytest.mark.parametrize("workflow", ["release.yml", "publish.yml"])
def test_every_workspace_distribution_is_offered_by_both_workflows(
    workflow: str,
) -> None:
    """A GitHub `type: choice` list can't read Python, so a test guards it.

    Adding a member to [tool.uv.workspace] members without adding it to both
    workflow dropdowns must fail here, naming the missing distribution.
    """
    offered = _package_choice_options(_WORKFLOWS_DIR / workflow)
    declared = set(workspace_manifests.workspace_packages(_REPO_ROOT))
    assert offered == declared, (
        f"{workflow} package.options is missing {declared - offered}"
        if declared - offered
        else f"{workflow} package.options declares extra unknown distributions "
        f"{offered - declared}"
    )


def test_release_and_publish_dropdowns_offer_the_same_distributions() -> None:
    """Declaring a distribution in one workflow's dropdown but not the other fails."""
    release_options = _package_choice_options(_WORKFLOWS_DIR / "release.yml")
    publish_options = _package_choice_options(_WORKFLOWS_DIR / "publish.yml")
    assert release_options == publish_options, (
        "release.yml and publish.yml package.options disagree: "
        f"only in release.yml: {release_options - publish_options}, "
        f"only in publish.yml: {publish_options - release_options}"
    )


def test_every_intra_workspace_requirement_declares_a_version_floor() -> None:
    unpinned = workspace_manifests.unpinned_requirements(_REPO_ROOT)
    assert not unpinned, (
        "intra-workspace dependencies must declare a lower bound so a real-PyPI "
        "install cannot resolve a stale sibling; unpinned: "
        + ", ".join(f"{r.consumer} -> {r.raw} ({r.origin})" for r in unpinned)
    )
