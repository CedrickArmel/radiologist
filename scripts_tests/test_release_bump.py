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


class TestParseReleaseBranchName:
    """`parse_release_branch_name` inverts `release_branch_name` for publish.yml."""

    def test_parses_member_distribution_branch_name(self):
        from release_bump import parse_release_branch_name

        assert parse_release_branch_name("release/radiologist-core-v0.2.0") == (
            "radiologist-core",
            "0.2.0",
        )

    def test_parses_root_distribution_branch_name(self):
        from release_bump import parse_release_branch_name

        assert parse_release_branch_name("release/radiologist-v0.3.1") == (
            "radiologist",
            "0.3.1",
        )

    def test_splits_on_the_last_dash_v_for_hyphenated_names(self):
        from release_bump import parse_release_branch_name

        assert parse_release_branch_name("release/radiologist-inference-v1.10.2") == (
            "radiologist-inference",
            "1.10.2",
        )

    def test_missing_release_prefix_raises_value_error(self):
        from release_bump import parse_release_branch_name

        with pytest.raises(ValueError):
            parse_release_branch_name("radiologist-core-v0.2.0")

    def test_unknown_distribution_raises_value_error(self):
        from release_bump import parse_release_branch_name

        with pytest.raises(ValueError):
            parse_release_branch_name("release/not-a-real-package-v0.2.0")

    def test_malformed_version_raises_value_error(self):
        from release_bump import parse_release_branch_name

        with pytest.raises(ValueError):
            parse_release_branch_name("release/radiologist-core-v0.2")

    def test_branch_without_v_marker_raises_value_error(self):
        from release_bump import parse_release_branch_name

        with pytest.raises(ValueError):
            parse_release_branch_name("release/radiologist-core")


class TestEnvironmentName:
    """`environment_name` maps a distribution to its GitHub Environment."""

    def test_member_distribution_environment_name(self):
        from release_bump import environment_name

        assert environment_name("radiologist-core") == "pypi-radiologist-core"

    def test_root_distribution_environment_name(self):
        from release_bump import environment_name

        assert environment_name("radiologist") == "pypi-radiologist"

    def test_unknown_distribution_raises_value_error(self):
        from release_bump import environment_name

        with pytest.raises(ValueError):
            environment_name("not-a-real-package")


class TestReleaseTag:
    """`release_tag` matches each distribution's configured commitizen tag_format."""

    def test_root_distribution_tag_has_no_suffix(self):
        from release_bump import release_tag

        assert release_tag("radiologist", "0.3.1") == "0.3.1"

    def test_member_distribution_tag_is_suffixed_with_its_name(self):
        from release_bump import release_tag

        assert release_tag("radiologist-core", "0.2.0") == "0.2.0-radiologist-core"

    def test_unknown_distribution_raises_value_error(self):
        from release_bump import release_tag

        with pytest.raises(ValueError):
            release_tag("not-a-real-package", "0.2.0")


class TestManifestVersion:
    """`manifest_version` reads `[project].version` at a given checkout."""

    def test_reads_member_distribution_version(self, tmp_path):
        from release_bump import manifest_version

        pkg_dir = tmp_path / "radiologist-core"
        pkg_dir.mkdir()
        (pkg_dir / "pyproject.toml").write_text(
            '[project]\nname = "radiologist-core"\nversion = "0.2.0"\n'
        )

        assert manifest_version(tmp_path, "radiologist-core") == "0.2.0"

    def test_reads_root_distribution_version_without_double_prefixing(self, tmp_path):
        from release_bump import manifest_version

        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "radiologist"\nversion = "0.3.1"\n'
        )

        assert manifest_version(tmp_path, "radiologist") == "0.3.1"

    def test_missing_manifest_raises_file_not_found_error(self, tmp_path):
        from release_bump import manifest_version

        with pytest.raises(FileNotFoundError):
            manifest_version(tmp_path, "radiologist-core")


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
