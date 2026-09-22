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

"""Behavioral tests for issue #227's release/publish distribution roster.

`release.yml`'s and `publish.yml`'s `workflow_dispatch.inputs.package.options`
are GitHub Actions `type: choice` dropdowns evaluated at workflow-definition
time -- they cannot read `scripts/release_bump.py`'s `PACKAGES` tuple, so
they stay hand-maintained literal lists. This module only pins down that
`radiologist-cli` (added to the workspace as a full member, see #227's issue
body) is present in both dropdowns, matching the derived `PACKAGES` roster.
The executable contract that *keeps* the three surfaces in sync is issue
#229's scope.
"""

import re
from pathlib import Path
from typing import List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

_OPTIONS_RE = re.compile(
    r"package:\n(?:.*\n)*?\s*options:\n((?:\s*-\s*\S+\n)+)", re.MULTILINE
)


def _package_options(workflow: str) -> List[str]:
    text = (_WORKFLOWS_DIR / workflow).read_text()
    match = _OPTIONS_RE.search(text)
    assert match, f"{workflow}: could not find package.options block"
    return [line.strip("- ").strip() for line in match.group(1).splitlines() if line]


@pytest.mark.parametrize("workflow", ["release.yml", "publish.yml"])
def test_package_options_include_radiologist_cli(workflow: str) -> None:
    assert "radiologist-cli" in _package_options(workflow)


@pytest.mark.parametrize("workflow", ["release.yml", "publish.yml"])
def test_package_options_match_the_release_bump_packages_roster(
    workflow: str,
) -> None:
    from release_bump import PACKAGES

    assert set(_package_options(workflow)) == set(PACKAGES)
