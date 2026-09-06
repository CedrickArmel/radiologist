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
``declared_version``.

Issue #229 completes the module with the requirement-parsing functions:
``intra_workspace_requirements``, ``unpinned_requirements``,
``stale_pin_floors`` and ``publishable_requirement_lines``.
``render_stale_pin_markdown`` stays unimplemented -- issue #233's scope --
and is still covered by ``test_workspace_manifests_skeleton.py``.
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


def _write_workspace(tmp_path: Path, members, root_extra_body: str = "") -> None:
    """Build a minimal fake workspace: root manifest plus one member per name."""
    members_toml = ", ".join(f'"{m}"' for m in members)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "radiologist"\n'
        'version = "0.1.0"\n'
        f"{root_extra_body}"
        "\n"
        "[tool.uv.workspace]\n"
        f"members = [{members_toml}]\n"
    )
    for member in members:
        member_dir = tmp_path / member
        member_dir.mkdir(exist_ok=True)
        if not (member_dir / "pyproject.toml").exists():
            (member_dir / "pyproject.toml").write_text(
                f'[project]\nname = "{member}"\nversion = "0.1.0"\n'
            )


class TestIntraWorkspaceRequirements:
    """`intra_workspace_requirements` finds edges naming workspace members."""

    def test_returns_only_requirements_naming_workspace_members(self, tmp_path):
        from workspace_manifests import intra_workspace_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=(
                'dependencies = ["radiologist-core>=0.1.0", "requests>=2.0.0"]\n'
            ),
        )

        result = intra_workspace_requirements(tmp_path, "radiologist")

        assert len(result) == 1
        assert result[0].target == "radiologist-core"
        assert result[0].consumer == "radiologist"

    def test_covers_default_dependencies_and_each_extra_in_declaration_order(
        self, tmp_path
    ):
        from workspace_manifests import intra_workspace_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core", "radiologist-cli"],
            root_extra_body=(
                'dependencies = ["radiologist-core[all]>=0.1.0"]\n'
                "\n"
                "[project.optional-dependencies]\n"
                'all = ["radiologist-cli[all]>=0.1.0"]\n'
            ),
        )

        result = intra_workspace_requirements(tmp_path, "radiologist")

        assert [(r.origin, r.target) for r in result] == [
            ("dependencies", "radiologist-core"),
            ("optional-dependencies.all", "radiologist-cli"),
        ]

    def test_extras_and_specifier_are_captured(self, tmp_path):
        from workspace_manifests import intra_workspace_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["radiologist-core[all]>=0.1.0"]\n'),
        )

        result = intra_workspace_requirements(tmp_path, "radiologist")

        assert result[0].extras == ("all",)
        assert result[0].specifier == ">=0.1.0"
        assert result[0].raw == "radiologist-core[all]>=0.1.0"

    def test_unpinned_requirement_has_empty_specifier(self, tmp_path):
        from workspace_manifests import intra_workspace_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["radiologist-core[all]"]\n'),
        )

        result = intra_workspace_requirements(tmp_path, "radiologist")

        assert result[0].specifier == ""

    def test_returns_empty_when_no_intra_workspace_requirement_declared(self, tmp_path):
        from workspace_manifests import intra_workspace_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["requests>=2.0.0"]\n'),
        )

        assert intra_workspace_requirements(tmp_path, "radiologist") == []


class TestUnpinnedRequirements:
    """`unpinned_requirements` finds every workspace-wide missing floor."""

    def test_returns_requirements_with_no_specifier(self, tmp_path):
        from workspace_manifests import unpinned_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["radiologist-core[all]"]\n'),
        )

        result = unpinned_requirements(tmp_path)

        assert len(result) == 1
        assert result[0].consumer == "radiologist"
        assert result[0].target == "radiologist-core"

    def test_empty_when_every_intra_workspace_requirement_is_pinned(self, tmp_path):
        from workspace_manifests import unpinned_requirements

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["radiologist-core[all]>=0.1.0"]\n'),
        )

        assert unpinned_requirements(tmp_path) == []

    def test_real_repository_has_no_unpinned_intra_workspace_requirement(self):
        from workspace_manifests import REPO_ROOT, unpinned_requirements

        unpinned = unpinned_requirements(REPO_ROOT)

        assert unpinned == [], [(r.consumer, r.raw, r.origin) for r in unpinned]


class TestStalePinFloors:
    """`stale_pin_floors` finds a pinned floor below its target's version."""

    def test_reports_requirement_whose_floor_is_below_target_version(self, tmp_path):
        from workspace_manifests import stale_pin_floors

        _write_workspace(tmp_path, ["radiologist-core"])
        (tmp_path / "radiologist-core" / "pyproject.toml").write_text(
            '[project]\nname = "radiologist-core"\nversion = "0.2.0"\n'
        )
        (tmp_path / "pyproject.toml").write_text(
            "[project]\n"
            'name = "radiologist"\n'
            'version = "0.1.0"\n'
            'dependencies = ["radiologist-core>=0.1.0"]\n'
            "\n"
            "[tool.uv.workspace]\n"
            'members = ["radiologist-core"]\n'
        )

        result = stale_pin_floors(tmp_path)

        assert len(result) == 1
        assert result[0].target == "radiologist-core"

    def test_does_not_report_requirement_whose_floor_equals_target_version(
        self, tmp_path
    ):
        from workspace_manifests import stale_pin_floors

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["radiologist-core>=0.1.0"]\n'),
        )

        assert stale_pin_floors(tmp_path) == []


class TestPublishableRequirementLines:
    """`publishable_requirement_lines` mirrors what a public index would see."""

    def test_returns_deduplicated_sorted_default_and_extra_requirements(self, tmp_path):
        from workspace_manifests import publishable_requirement_lines

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=(
                'dependencies = ["radiologist-core[all]>=0.1.0", "requests>=2.0.0"]\n'
                "\n"
                "[project.optional-dependencies]\n"
                'extra-a = ["requests>=2.0.0"]\n'
                'extra-b = ["click>=8.0.0"]\n'
            ),
        )

        result = publishable_requirement_lines(tmp_path, "radiologist")

        assert result == sorted(
            {
                "radiologist-core[all]>=0.1.0",
                "requests>=2.0.0",
                "click>=8.0.0",
            }
        )

    def test_returns_empty_for_a_distribution_with_no_dependency_at_all(self, tmp_path):
        from workspace_manifests import publishable_requirement_lines

        _write_workspace(tmp_path, ["radiologist-core"])

        assert publishable_requirement_lines(tmp_path, "radiologist-core") == []

    def test_does_not_apply_uv_workspace_source_rewriting(self, tmp_path):
        from workspace_manifests import publishable_requirement_lines

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=(
                'dependencies = ["radiologist-core[all]>=0.1.0"]\n'
                "\n"
                "[tool.uv.sources]\n"
                "radiologist-core = { workspace = true }\n"
            ),
        )

        result = publishable_requirement_lines(tmp_path, "radiologist")

        assert result == ["radiologist-core[all]>=0.1.0"]

    def test_does_not_apply_tool_uv_constraint_dependencies(self, tmp_path):
        from workspace_manifests import publishable_requirement_lines

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=(
                'dependencies = ["radiologist-core[all]>=0.1.0"]\n'
                "\n"
                "[tool.uv]\n"
                'constraint-dependencies = ["fastapi<0.116.0"]\n'
            ),
        )

        result = publishable_requirement_lines(tmp_path, "radiologist")

        assert result == ["radiologist-core[all]>=0.1.0"]

    def test_real_repository_preserves_extras_on_publishable_lines(self):
        from workspace_manifests import REPO_ROOT, publishable_requirement_lines

        result = publishable_requirement_lines(REPO_ROOT, "radiologist")

        assert any(line.startswith("radiologist-core[all]") for line in result)


class TestRequirementLinesCli:
    """The ``requirement-lines`` CLI subcommand (issue #232's resolution guard)."""

    def test_prints_one_requirement_per_line(self, tmp_path, capsys):
        from workspace_manifests import _main

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=('dependencies = ["radiologist-core[all]>=0.1.0"]\n'),
        )

        exit_code = _main(
            [
                "requirement-lines",
                "--repo-root",
                str(tmp_path),
                "--package",
                "radiologist",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out.splitlines() == ["radiologist-core[all]>=0.1.0"]

    def test_prints_nothing_for_a_distribution_with_no_dependency(
        self, tmp_path, capsys
    ):
        from workspace_manifests import _main

        _write_workspace(tmp_path, ["radiologist-core"])

        exit_code = _main(
            [
                "requirement-lines",
                "--repo-root",
                str(tmp_path),
                "--package",
                "radiologist-core",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out == ""

    def test_prints_every_extra_alongside_default_dependencies(self, tmp_path, capsys):
        from workspace_manifests import _main

        _write_workspace(
            tmp_path,
            ["radiologist-core"],
            root_extra_body=(
                'dependencies = ["radiologist-core[all]>=0.1.0"]\n'
                "\n"
                "[project.optional-dependencies]\n"
                'extra-a = ["requests>=2.0.0"]\n'
            ),
        )

        exit_code = _main(
            [
                "requirement-lines",
                "--repo-root",
                str(tmp_path),
                "--package",
                "radiologist",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert captured.out.splitlines() == [
            "radiologist-core[all]>=0.1.0",
            "requests>=2.0.0",
        ]
