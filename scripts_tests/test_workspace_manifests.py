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

"""Behavioral tests for the workspace-roster readers in ``workspace_manifests``.

Issue #227 brings ``radiologist-cli`` onto the release/publish surface by
making ``scripts/release_bump.py``'s ``PACKAGES`` roster a live read of the
root manifest's ``[tool.uv.workspace] members`` instead of a hand-maintained
tuple. These tests cover the four reader functions that make that possible:
``workspace_packages``, ``package_dir``, ``package_manifest_path`` and
``declared_version``. The remaining functions in this module stay
unimplemented (issue #229's scope) and are covered by
``test_workspace_manifests_skeleton.py``.
"""

from pathlib import Path

import pytest


def _write_root_manifest(repo_root: Path, members) -> None:
    members_toml = ", ".join(f'"{m}"' for m in members)
    (repo_root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "radiologist"\n'
        'version = "0.1.0"\n'
        "\n"
        "[tool.uv.workspace]\n"
        f"members = [{members_toml}]\n"
    )


class TestWorkspacePackages:
    """`workspace_packages` reads the root distribution followed by its members."""

    def test_returns_root_distribution_followed_by_declared_members_in_order(
        self, tmp_path
    ):
        from workspace_manifests import workspace_packages

        _write_root_manifest(tmp_path, ["radiologist-core", "radiologist-cli"])

        assert workspace_packages(tmp_path) == (
            "radiologist",
            "radiologist-core",
            "radiologist-cli",
        )

    def test_result_changes_when_the_root_manifests_member_list_changes(self, tmp_path):
        from workspace_manifests import workspace_packages

        _write_root_manifest(tmp_path, ["radiologist-core"])
        before = workspace_packages(tmp_path)

        _write_root_manifest(tmp_path, ["radiologist-core", "radiologist-cli"])
        after = workspace_packages(tmp_path)

        assert before == ("radiologist", "radiologist-core")
        assert after == ("radiologist", "radiologist-core", "radiologist-cli")

    def test_missing_workspace_members_key_raises_key_error(self, tmp_path):
        from workspace_manifests import workspace_packages

        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "radiologist"\nversion = "0.1.0"\n'
        )

        with pytest.raises(KeyError):
            workspace_packages(tmp_path)

    def test_real_repository_declares_radiologist_cli(self):
        from workspace_manifests import REPO_ROOT, workspace_packages

        assert "radiologist-cli" in workspace_packages(REPO_ROOT)


class TestPackageDir:
    """`package_dir` maps a declared distribution to its on-disk directory."""

    def test_root_distribution_maps_to_repository_root(self, tmp_path):
        from workspace_manifests import package_dir

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        assert package_dir(tmp_path, "radiologist") == "."

    def test_member_distribution_maps_to_its_own_directory_name(self, tmp_path):
        from workspace_manifests import package_dir

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        assert package_dir(tmp_path, "radiologist-cli") == "radiologist-cli"

    def test_unknown_distribution_raises_value_error(self, tmp_path):
        from workspace_manifests import package_dir

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        with pytest.raises(ValueError):
            package_dir(tmp_path, "not-a-real-package")


class TestPackageManifestPath:
    """`package_manifest_path` locates a distribution's `pyproject.toml`."""

    def test_root_distribution_manifest_is_at_repository_root(self, tmp_path):
        from workspace_manifests import package_manifest_path

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        assert package_manifest_path(tmp_path, "radiologist") == (
            tmp_path / "pyproject.toml"
        )

    def test_member_distribution_manifest_is_under_its_own_directory(self, tmp_path):
        from workspace_manifests import package_manifest_path

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        assert package_manifest_path(tmp_path, "radiologist-cli") == (
            tmp_path / "radiologist-cli" / "pyproject.toml"
        )

    def test_unknown_distribution_raises_value_error(self, tmp_path):
        from workspace_manifests import package_manifest_path

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        with pytest.raises(ValueError):
            package_manifest_path(tmp_path, "not-a-real-package")


class TestDeclaredVersion:
    """`declared_version` reads `[project].version` from a distribution's manifest."""

    def test_reads_member_distributions_version(self, tmp_path):
        from workspace_manifests import declared_version

        _write_root_manifest(tmp_path, ["radiologist-cli"])
        cli_dir = tmp_path / "radiologist-cli"
        cli_dir.mkdir()
        (cli_dir / "pyproject.toml").write_text(
            '[project]\nname = "radiologist-cli"\nversion = "0.1.0"\n'
        )

        assert declared_version(tmp_path, "radiologist-cli") == "0.1.0"

    def test_reads_root_distributions_version(self, tmp_path):
        from workspace_manifests import declared_version

        _write_root_manifest(tmp_path, ["radiologist-cli"])

        assert declared_version(tmp_path, "radiologist") == "0.1.0"
