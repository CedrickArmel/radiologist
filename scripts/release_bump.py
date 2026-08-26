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

"""Helper logic for ``.github/workflows/release.yml``.

The workflow triggers a ``cz bump --files-only`` for one of the six
``radiologist`` distributions and then lands the resulting file changes as a
single GitHub-signed commit via the GraphQL ``createCommitOnBranch``
mutation. That mutation requires every changed file's contents to be
base64-encoded and bundled into one payload — easy to get subtly wrong in
bash. This module isolates that logic so it can be unit tested, and exposes
a small CLI so workflow steps can call it directly.

Only pure, testable logic lives here: mapping a distribution name to its
``cz`` working directory, naming the release branch, listing the files a
bump touches, encoding them, and building the mutation payload. Talking to
GitHub (creating the branch ref, calling the GraphQL API, opening the pull
request) stays in the workflow YAML via the ``gh`` CLI — a true process
boundary this module does not own.
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Dict, List

PACKAGES = (
    "radiologist",
    "radiologist-utils",
    "radiologist-etl",
    "radiologist-core",
    "radiologist-inference",
    "radiologist-registry",
)

_ROOT_PACKAGE = "radiologist"


def package_dir(package: str) -> str:
    """Return the ``cz`` working directory for ``package``.

    The root meta-package versions at the repository root (``.``); each of
    the five workspace members versions inside its own directory, which
    shares its name.
    """
    if package not in PACKAGES:
        raise ValueError(
            f"Unknown distribution {package!r}; expected one of {PACKAGES}"
        )
    return "." if package == _ROOT_PACKAGE else package


def release_branch_name(package: str, version: str) -> str:
    """Return the ``release/<package>-v<version>`` head branch name.

    This is the frozen convention issue #165 parses to resolve which
    distribution and version a merged release pull request published.
    """
    return f"release/{package}-v{version}"


def changed_relative_paths(package: str) -> List[str]:
    """List the repo-root-relative paths a bump of ``package`` touches.

    Every bump rewrites that distribution's ``pyproject.toml`` and
    ``CHANGELOG.md``, plus the workspace-wide ``uv.lock`` that the ``uv``
    commitizen version provider refreshes as a side effect.
    """
    directory = package_dir(package)
    if directory == ".":
        return ["pyproject.toml", "CHANGELOG.md", "uv.lock"]
    return [f"{directory}/pyproject.toml", f"{directory}/CHANGELOG.md", "uv.lock"]


def encode_file_changes(repo_root: Path, package: str) -> List[Dict[str, str]]:
    """Read and base64-encode each file a bump of ``package`` changed.

    Returns a list of ``{"path": ..., "contents": ...}`` entries ready to
    drop into the GraphQL ``createCommitOnBranch`` mutation's
    ``fileChanges.additions``.
    """
    additions = []
    for relative_path in changed_relative_paths(package):
        file_path = Path(repo_root) / relative_path
        contents = file_path.read_bytes()
        additions.append(
            {
                "path": relative_path,
                "contents": base64.b64encode(contents).decode("ascii"),
            }
        )
    return additions


def commit_headline(package: str, old_version: str, new_version: str) -> str:
    """Return the single-commit headline, matching each package's ``bump_message``."""
    return f"bump: {package} {old_version} → {new_version}"


def build_commit_mutation_variables(
    repository_name_with_owner: str,
    branch_name: str,
    expected_head_oid: str,
    headline: str,
    additions: List[Dict[str, str]],
) -> Dict:
    """Assemble the ``createCommitOnBranch`` mutation's ``input`` variables.

    ``expectedHeadOid`` is set to the branch head created just before this
    call, so a concurrent write to that branch fails the mutation loudly
    instead of silently overwriting it.
    """
    return {
        "input": {
            "branch": {
                "repositoryNameWithOwner": repository_name_with_owner,
                "branchName": branch_name,
            },
            "message": {
                "headline": headline,
            },
            "expectedHeadOid": expected_head_oid,
            "fileChanges": {
                "additions": additions,
            },
        }
    }


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dir_parser = subparsers.add_parser("package-dir")
    dir_parser.add_argument("package")

    branch_parser = subparsers.add_parser("branch-name")
    branch_parser.add_argument("package")
    branch_parser.add_argument("version")

    headline_parser = subparsers.add_parser("commit-headline")
    headline_parser.add_argument("package")
    headline_parser.add_argument("old_version")
    headline_parser.add_argument("new_version")

    mutation_parser = subparsers.add_parser("mutation-variables")
    mutation_parser.add_argument("--repo-root", required=True)
    mutation_parser.add_argument("--package", required=True)
    mutation_parser.add_argument("--repository-name-with-owner", required=True)
    mutation_parser.add_argument("--branch-name", required=True)
    mutation_parser.add_argument("--expected-head-oid", required=True)
    mutation_parser.add_argument("--headline", required=True)

    args = parser.parse_args(argv)

    if args.command == "package-dir":
        print(package_dir(args.package))
    elif args.command == "branch-name":
        print(release_branch_name(args.package, args.version))
    elif args.command == "commit-headline":
        print(commit_headline(args.package, args.old_version, args.new_version))
    elif args.command == "mutation-variables":
        additions = encode_file_changes(Path(args.repo_root), args.package)
        variables = build_commit_mutation_variables(
            repository_name_with_owner=args.repository_name_with_owner,
            branch_name=args.branch_name,
            expected_head_oid=args.expected_head_oid,
            headline=args.headline,
            additions=additions,
        )
        print(json.dumps(variables))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
