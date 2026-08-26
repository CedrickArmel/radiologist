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

"""Behavioral tests for the release-bump helper used by ``release.yml``.

These functions back the parts of the release workflow that are too
error-prone to hand-roll in YAML/bash: mapping a distribution name to its
``cz`` working directory, naming the release branch, and building the
GraphQL ``createCommitOnBranch`` payload (which requires base64-encoding
every changed file into a single mutation).
"""

import base64
import json

import pytest


class TestPackageDir:
    """`package_dir` maps a distribution name to its `cz` working directory."""

    def test_root_meta_package_maps_to_repository_root(self):
        from release_bump import package_dir

        assert package_dir("radiologist") == "."

    def test_member_distribution_maps_to_its_own_directory(self):
        from release_bump import package_dir

        assert package_dir("radiologist-core") == "radiologist-core"

    def test_unknown_distribution_raises_value_error(self):
        from release_bump import package_dir

        with pytest.raises(ValueError):
            package_dir("not-a-real-package")


class TestReleaseBranchName:
    """`release_branch_name` produces the `release/<pkg>-v<version>` format."""

    def test_formats_member_distribution_branch_name(self):
        from release_bump import release_branch_name

        assert (
            release_branch_name("radiologist-core", "0.2.0")
            == "release/radiologist-core-v0.2.0"
        )

    def test_formats_root_distribution_branch_name(self):
        from release_bump import release_branch_name

        assert (
            release_branch_name("radiologist", "0.3.1") == "release/radiologist-v0.3.1"
        )


class TestChangedRelativePaths:
    """`changed_relative_paths` lists the files a bump touches, repo-root-relative."""

    def test_member_distribution_includes_its_own_pyproject_and_changelog_and_lock(
        self,
    ):
        from release_bump import changed_relative_paths

        paths = changed_relative_paths("radiologist-core")

        assert paths == [
            "radiologist-core/pyproject.toml",
            "radiologist-core/CHANGELOG.md",
            "uv.lock",
        ]

    def test_root_distribution_paths_are_not_double_prefixed(self):
        from release_bump import changed_relative_paths

        paths = changed_relative_paths("radiologist")

        assert paths == ["pyproject.toml", "CHANGELOG.md", "uv.lock"]


class TestEncodeFileChanges:
    """`encode_file_changes` builds base64 GraphQL additions from real files."""

    def test_reads_and_base64_encodes_each_changed_file(self, tmp_path):
        from release_bump import encode_file_changes

        pkg_dir = tmp_path / "radiologist-core"
        pkg_dir.mkdir()
        (pkg_dir / "pyproject.toml").write_bytes(b'[project]\nversion = "0.2.0"\n')
        (pkg_dir / "CHANGELOG.md").write_bytes(b"## 0.2.0\n")
        (tmp_path / "uv.lock").write_bytes(b"version = 1\n")

        additions = encode_file_changes(tmp_path, "radiologist-core")

        assert additions == [
            {
                "path": "radiologist-core/pyproject.toml",
                "contents": base64.b64encode(b'[project]\nversion = "0.2.0"\n').decode(
                    "ascii"
                ),
            },
            {
                "path": "radiologist-core/CHANGELOG.md",
                "contents": base64.b64encode(b"## 0.2.0\n").decode("ascii"),
            },
            {
                "path": "uv.lock",
                "contents": base64.b64encode(b"version = 1\n").decode("ascii"),
            },
        ]

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        from release_bump import encode_file_changes

        with pytest.raises(FileNotFoundError):
            encode_file_changes(tmp_path, "radiologist-core")


class TestCommitHeadline:
    """`commit_headline` matches each package's configured `bump_message` shape."""

    def test_member_distribution_headline(self):
        from release_bump import commit_headline

        assert (
            commit_headline("radiologist-core", "0.1.0", "0.2.0")
            == "bump: radiologist-core 0.1.0 → 0.2.0"
        )

    def test_root_distribution_headline(self):
        from release_bump import commit_headline

        assert (
            commit_headline("radiologist", "0.1.0", "0.3.1")
            == "bump: radiologist 0.1.0 → 0.3.1"
        )


class TestBuildCommitMutationVariables:
    """`build_commit_mutation_variables` assembles the createCommitOnBranch input."""

    def test_builds_graphql_input_with_expected_head_oid_and_additions(self):
        from release_bump import build_commit_mutation_variables

        additions = [{"path": "uv.lock", "contents": "dGVzdA=="}]

        variables = build_commit_mutation_variables(
            repository_name_with_owner="CedrickArmel/radiologist",
            branch_name="release/radiologist-core-v0.2.0",
            expected_head_oid="abc123",
            headline="bump: radiologist-core 0.1.0 → 0.2.0",
            additions=additions,
        )

        assert variables == {
            "input": {
                "branch": {
                    "repositoryNameWithOwner": "CedrickArmel/radiologist",
                    "branchName": "release/radiologist-core-v0.2.0",
                },
                "message": {
                    "headline": "bump: radiologist-core 0.1.0 → 0.2.0",
                },
                "expectedHeadOid": "abc123",
                "fileChanges": {
                    "additions": additions,
                },
            }
        }

    def test_output_is_json_serializable(self):
        from release_bump import build_commit_mutation_variables

        variables = build_commit_mutation_variables(
            repository_name_with_owner="CedrickArmel/radiologist",
            branch_name="release/radiologist-v0.1.1",
            expected_head_oid="deadbeef",
            headline="bump: radiologist 0.1.0 → 0.1.1",
            additions=[],
        )

        json.dumps(variables)  # must not raise
