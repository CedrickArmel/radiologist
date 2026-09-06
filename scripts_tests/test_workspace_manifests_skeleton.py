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

"""Structural (shape-only) contract tests for issue #225's skeleton.

Issue #225 stubs the entire "harden CI/build/release" epic surface (milestone
#20) so slices #226-#234 can start in parallel. It changes no observable
behavior: every new function body is ``raise NotImplementedError``. These
tests therefore assert only what a skeleton commits to -- the module and its
public names exist, are importable, and carry the exact signatures the
epic's slice issues will call -- not what those functions compute. See the
epic spec's "Configuration/interface contract" section for the frozen
contract these tests pin down.

Issue #227 has since implemented ``workspace_packages``, ``package_dir``,
``package_manifest_path`` and ``declared_version`` (see
``test_workspace_manifests.py`` for their behavioral coverage) so the roster
``scripts/release_bump.py`` exposes can be derived from the workspace
manifest instead of hand-maintained. The remaining functions --
``intra_workspace_requirements``, ``unpinned_requirements``,
``stale_pin_floors``, ``publishable_requirement_lines`` and
``render_stale_pin_markdown`` -- are issue #229's scope and stay
unimplemented here.

``.github/actions/setup-uv/action.yml`` is a plain composite action file, no
Python involved, so its own shape is asserted with the workflow-as-text
pattern already used by ``test_ci_workflows_exclude_ray.py`` -- no
``pyyaml`` dependency.
"""

import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SETUP_UV_ACTION = _REPO_ROOT / ".github" / "actions" / "setup-uv" / "action.yml"


class TestWorkspaceManifestsModuleShape:
    """``scripts/workspace_manifests.py`` importable, with the frozen contract."""

    def test_module_imports_cleanly(self) -> None:
        import workspace_manifests  # noqa: F401

    def test_module_exposes_repo_root_and_root_package_constants(self) -> None:
        import workspace_manifests

        assert workspace_manifests.REPO_ROOT == _REPO_ROOT
        assert workspace_manifests.ROOT_PACKAGE == "radiologist"

    def test_intra_workspace_requirement_dataclass_has_the_frozen_fields(
        self,
    ) -> None:
        import workspace_manifests

        fields = workspace_manifests.IntraWorkspaceRequirement.__dataclass_fields__
        assert set(fields) == {
            "consumer",
            "target",
            "extras",
            "specifier",
            "origin",
            "raw",
        }

    def test_public_functions_exist_with_the_declared_parameter_names(self) -> None:
        import workspace_manifests

        expected_params = {
            "workspace_packages": ["repo_root"],
            "package_dir": ["repo_root", "package"],
            "package_manifest_path": ["repo_root", "package"],
            "declared_version": ["repo_root", "package"],
            "intra_workspace_requirements": ["repo_root", "package"],
            "unpinned_requirements": ["repo_root"],
            "stale_pin_floors": ["repo_root"],
            "publishable_requirement_lines": ["repo_root", "package"],
            "render_stale_pin_markdown": ["findings", "target_versions"],
        }
        for name, params in expected_params.items():
            func = getattr(workspace_manifests, name)
            assert list(inspect.signature(func).parameters) == params, name

    def test_every_still_unimplemented_public_function_raises_not_implemented_error(
        self,
    ) -> None:
        """``workspace_packages``/``package_dir``/``package_manifest_path``/
        ``declared_version`` are implemented (issue #227) and covered
        behaviorally in ``test_workspace_manifests.py`` instead.
        """
        import workspace_manifests

        with pytest.raises(NotImplementedError):
            workspace_manifests.intra_workspace_requirements(_REPO_ROOT, "radiologist")
        with pytest.raises(NotImplementedError):
            workspace_manifests.unpinned_requirements(_REPO_ROOT)
        with pytest.raises(NotImplementedError):
            workspace_manifests.stale_pin_floors(_REPO_ROOT)
        with pytest.raises(NotImplementedError):
            workspace_manifests.publishable_requirement_lines(_REPO_ROOT, "radiologist")
        with pytest.raises(NotImplementedError):
            workspace_manifests.render_stale_pin_markdown([], {})

    def test_main_entrypoint_exists(self) -> None:
        import workspace_manifests

        assert callable(workspace_manifests._main)


class TestSetupUvCompositeActionShape:
    """``.github/actions/setup-uv/action.yml`` exists and is inert."""

    def test_action_file_exists(self) -> None:
        assert _SETUP_UV_ACTION.is_file()

    def test_action_declares_composite_runs_and_expected_inputs(self) -> None:
        text = _SETUP_UV_ACTION.read_text()
        assert "using: composite" in text
        for input_name in (
            "python-version",
            "setup-python",
            "enable-cache",
            "locked",
        ):
            assert f"{input_name}:" in text, input_name

    def test_action_performs_no_checkout_no_sync_no_test_run(self) -> None:
        text = _SETUP_UV_ACTION.read_text()
        assert "uses: actions/checkout" not in text
        assert "run: uv sync" not in text
        assert "run: make test" not in text

    def test_action_documents_uv_locked_over_uv_frozen(self) -> None:
        text = _SETUP_UV_ACTION.read_text()
        assert "UV_LOCKED" in text
        assert "UV_FROZEN" in text  # documented as deliberately NOT used

    def test_existing_setup_and_test_action_is_untouched(self) -> None:
        setup_and_test = (
            _REPO_ROOT / ".github" / "actions" / "setup-and-test" / "action.yml"
        )
        text = setup_and_test.read_text()
        # setup-and-test still owns checkout + sync + make test; this
        # skeleton must not have folded that responsibility away.
        assert "actions/checkout" in text
        assert "uv sync" in text
        assert "make test" in text
