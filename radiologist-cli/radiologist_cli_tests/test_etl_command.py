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

"""Behavioral tests for the ``radiologist etl`` command group's three
subcommands (``extract``, ``assign-split``, ``build``), replacing the
retired single-command monolithic flow.

Dispatch-only scenarios (no subcommand, unknown subcommand, group-level
``--help``) short-circuit before any Hydra composition happens, so they run
in-process. Real per-stage runs exercise the real ``radiologist.etl``
business logic in-process with Prefect's own orchestration/tracking HTTP
calls stubbed out (a true process boundary — see
``radiologist-etl/radiologist_etl_tests/test_prefect_pipelines.py`` for the
same technique and its rationale): this environment has no reachable
Prefect API (the local ephemeral server is broken by a Starlette/Prefect
version mismatch unrelated to this change, and the configured Prefect Cloud
credentials must not be hit by tests).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pytest
from PIL import Image

import radiologist.cli.groups.etl as etl_group
import radiologist.etl.prefect_pipelines as prefect_pipelines
from radiologist.etl import records_reader

CLI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLI_ROOT.parent
_SRC_DIRS = [
    str(REPO_ROOT / pkg / "src")
    for pkg in (
        "radiologist-cli",
        "radiologist-etl",
        "radiologist-utils",
        "radiologist-inference",
        "radiologist-registry",
    )
    if (REPO_ROOT / pkg / "src").is_dir()
]


def _run_cli_subprocess(argv: List[str]) -> Tuple[int, str, str]:
    env = dict(os.environ)
    env.pop("PREFECT_API_URL", None)
    env.pop("PREFECT_API_KEY", None)
    # Force the subprocess to import this checkout's sources rather than
    # whatever an editable install's .pth files point at -- load-bearing
    # when this test suite runs from a git worktree.
    env["PYTHONPATH"] = os.pathsep.join(_SRC_DIRS + [env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from radiologist.cli.groups.etl import run\n"
            "sys.exit(run(sys.argv[1:]))\n",
            *argv,
        ],
        cwd=CLI_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _build_image_tree(root: Path, n_per_class: int = 2) -> Path:
    images = root / "images"
    rng = np.random.default_rng(7)
    for label in ("NORMAL", "ABNORMAL"):
        for i in range(n_per_class):
            _write_png(
                images / label / f"img{i:03d}.png",
                rng.integers(0, 256, (10, 10, 3), dtype=np.uint8),
            )
    return images


def _write_listing(tmp_path: Path, paths: List[str], name: str = "listing.txt") -> Path:
    listing_path = tmp_path / name
    listing_path.write_text("\n".join(paths) + "\n")
    return listing_path


def _all_image_paths(image_dir: Path) -> List[str]:
    return [str(p) for p in sorted(image_dir.rglob("*.png"))]


def _make_extract_manifest(
    tmp_path: Path, image_dir: Path, manifests_dir: Path
) -> Path:
    from radiologist.etl.manifest import JsonlWriter, ManifestRecord

    records = [
        ManifestRecord(
            manifest_id="extractrun000001",
            path=str(p),
            filename=p.name,
            label=p.parent.name,
            split="",
            stats={},
        )
        for p in sorted(image_dir.rglob("*.png"))
    ]
    manifest_path = manifests_dir / "extract-extractrun000001.jsonl"
    JsonlWriter().write(records, str(manifest_path))
    return manifest_path


def _make_split_manifest(tmp_path: Path, image_dir: Path, dest: Path) -> Path:
    from radiologist.etl.manifest import JsonlWriter, ManifestRecord

    records = [
        ManifestRecord(
            manifest_id="assignrun0000001",
            path=str(p),
            filename=p.name,
            label=p.parent.name,
            split="train",
            stats={},
        )
        for p in sorted(image_dir.rglob("*.png"))
    ]
    manifest_path = dest / "manifest-assignrun0000001.jsonl"
    JsonlWriter().write(records, str(manifest_path))
    return manifest_path


class _FlowSpy:
    """Stand-in for a Prefect ``Flow`` object exposing ``.with_options()`` and
    a direct call, wired to the real flow's own ``.fn`` business logic —
    sidesteps this environment's unreachable Prefect orchestration engine
    (see ``radiologist-etl/radiologist_etl_tests/test_prefect_pipelines.py``'s
    ``_FakeFlow`` and its module docstring for the same rationale) while
    every stage still runs for real.
    """

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def with_options(self, **_kwargs: object) -> "_FlowSpy":
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.fn(*args, **kwargs)


@pytest.fixture()
def bypass_prefect_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Prefect's HTTP orchestration/artifact calls (a true process
    boundary), leaving all real ETL business logic untouched."""
    monkeypatch.setattr(prefect_pipelines, "create_link_artifact", lambda **_: None)
    monkeypatch.setattr(prefect_pipelines, "create_markdown_artifact", lambda **_: None)
    monkeypatch.setattr(prefect_pipelines, "create_table_artifact", lambda **_: None)
    monkeypatch.setattr(
        prefect_pipelines, "extract_flow", _FlowSpy(prefect_pipelines.extract_flow.fn)
    )
    monkeypatch.setattr(
        prefect_pipelines,
        "assign_split_flow",
        _FlowSpy(prefect_pipelines.assign_split_flow.fn),
    )
    monkeypatch.setattr(
        prefect_pipelines, "build_flow", _FlowSpy(prefect_pipelines.build_flow.fn)
    )


@pytest.fixture()
def clear_global_hydra():
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()


# --- dispatch: no/unknown subcommand, help -----------------------------------


def test_no_subcommand_prints_usage_naming_subcommands_and_exits_nonzero(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = etl_group.run([])

    captured = capsys.readouterr()
    assert exit_code != 0
    for subcommand in etl_group.SUBCOMMANDS:
        assert subcommand in captured.err


def test_unknown_subcommand_prints_usage_naming_subcommands_and_exits_nonzero(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = etl_group.run(["frobnicate"])

    captured = capsys.readouterr()
    assert exit_code != 0
    for subcommand in etl_group.SUBCOMMANDS:
        assert subcommand in captured.err


def test_help_flag_prints_usage_naming_subcommands_and_exits_zero(
    capsys: pytest.CaptureFixture,
) -> None:
    exit_code = etl_group.run(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    for subcommand in etl_group.SUBCOMMANDS:
        assert subcommand in captured.out


# --- missing input existence checks ------------------------------------------


def test_extract_missing_file_list_exits_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"
    returncode, _, stderr = _run_cli_subprocess(
        [
            "extract",
            f"file_list={missing}",
            f"destination={tmp_path / 'out'}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    assert returncode == 2
    assert "Error: " in stderr
    assert str(missing) in stderr


def test_assign_split_missing_manifests_dir_exits_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "no-manifests-here"
    returncode, _, stderr = _run_cli_subprocess(
        [
            "assign-split",
            f"manifests_dir={missing}",
            f"destination={tmp_path / 'out'}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    assert returncode == 2
    assert "Error: " in stderr
    assert str(missing) in stderr


def test_build_missing_split_manifest_exits_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-manifest.jsonl"
    returncode, _, stderr = _run_cli_subprocess(
        [
            "build",
            f"split_manifest={missing}",
            f"shard_root={tmp_path / 'shards'}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    assert returncode == 2
    assert "Error: " in stderr
    assert str(missing) in stderr


# --- extract subcommand -------------------------------------------------------


def test_extract_subcommand_prints_record_with_run_id_and_counts(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    paths = _all_image_paths(images)
    listing = _write_listing(tmp_path, paths)

    exit_code = etl_group.run(
        [
            "extract",
            "--output=json",
            f"file_list={listing}",
            f"destination={tmp_path / 'out'}",
            "workers=1",
            "masks_root=null",
            "iqr_columns=[]",
            # local.yaml's default ProcessPoolTaskRunner needs a live
            # Prefect flow run to map tasks under; unreachable in this
            # environment (see module docstring) -- drop the runner group
            # so this real end-to-end run uses extract()'s own local
            # default mapper instead.
            "~runner",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert set(payload) == {
        "run_id",
        "manifest_path",
        "total",
        "succeeded",
        "failed",
        "excluded",
    }
    assert payload["total"] == len(paths)
    assert payload["succeeded"] == len(paths)
    assert payload["failed"] == 0


def test_extract_subcommand_applies_config_overrides(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    """``max_failure_rate`` override takes effect: with one unreadable path
    in the listing and the threshold left at zero, the stage's own real
    failure-rate check raises, proving the override reached the real stage
    rather than being silently ignored."""
    images = _build_image_tree(tmp_path, n_per_class=1)
    paths = _all_image_paths(images) + [str(tmp_path / "missing.png")]
    listing = _write_listing(tmp_path, paths)

    exit_code = etl_group.run(
        [
            "extract",
            f"file_list={listing}",
            f"destination={tmp_path / 'out'}",
            "workers=1",
            "masks_root=null",
            "iqr_columns=[]",
            "max_failure_rate=0.0",
            "~runner",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "Error: " in captured.err
    assert "failure rate" in captured.err


def test_extract_runner_override_selects_the_runner_family(
    tmp_path: Path,
    clear_global_hydra: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves ``run_extract``'s own wiring -- that a ``runner=`` override
    reaches ``resolve_execution`` with the matching family -- the same way
    ``radiologist-etl/radiologist_etl_tests/test_prefect_pipelines.py``'s
    ``test_run_extract_resolves_the_plan_and_attaches_its_task_runner``
    does: the flow body itself is replaced with a no-op, because a real
    attached task runner needs mapped-task execution under a live Prefect
    engine (unreachable in this environment -- see that file's module
    docstring), and this test's only job is to check which family gets
    selected, not to re-run extract-stage business logic.
    """
    from radiologist.etl.models import ExtractResult

    images = _build_image_tree(tmp_path)
    paths = _all_image_paths(images)
    listing = _write_listing(tmp_path, paths)

    real_resolve = prefect_pipelines.resolve_execution
    seen_families: List[str] = []

    def _spy(runner_cfg, batch_size=None):
        plan = real_resolve(runner_cfg, batch_size=batch_size)
        seen_families.append(plan.family)
        return plan

    monkeypatch.setattr(prefect_pipelines, "resolve_execution", _spy)
    fake_result = ExtractResult(
        run_id="r",
        manifest_path="m",
        total=0,
        succeeded=0,
        failed=0,
        failure_rate=0.0,
        excluded=0,
    )
    monkeypatch.setattr(
        prefect_pipelines,
        "extract_flow",
        _FlowSpy(lambda cfg, execution=None: fake_result),
    )

    exit_code = etl_group.run(
        [
            "extract",
            f"file_list={listing}",
            f"destination={tmp_path / 'out'}",
            "workers=1",
            "masks_root=null",
            "iqr_columns=[]",
            "runner=local",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    assert exit_code == 0
    assert seen_families == ["local"]


def test_extract_with_uninstalled_runner_backend_exits_nonzero_naming_extra(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    paths = _all_image_paths(images)
    listing = _write_listing(tmp_path, paths)

    exit_code = etl_group.run(
        [
            "extract",
            f"file_list={listing}",
            f"destination={tmp_path / 'out'}",
            "masks_root=null",
            "iqr_columns=[]",
            "runner=ray_local",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "radiologist-etl[ray]" in captured.err


# --- assign-split subcommand --------------------------------------------------


def test_assign_split_subcommand_prints_record_with_run_id_and_counts(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    manifests_dir = tmp_path / "manifests"
    _make_extract_manifest(tmp_path, images, manifests_dir)

    exit_code = etl_group.run(
        [
            "assign-split",
            "--output=json",
            f"manifests_dir={manifests_dir}",
            f"destination={tmp_path / 'dest'}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert set(payload) == {
        "run_id",
        "split_manifest_path",
        "source_manifest_count",
        "record_count",
        "duplicate_count",
    }
    assert payload["source_manifest_count"] == 1
    assert payload["record_count"] == 4


def test_assign_split_subcommand_applies_config_overrides(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    manifests_dir = tmp_path / "manifests"
    _make_extract_manifest(tmp_path, images, manifests_dir)

    exit_code = etl_group.run(
        [
            "assign-split",
            "--output=json",
            f"manifests_dir={manifests_dir}",
            f"destination={tmp_path / 'dest'}",
            'split_ratios=[["train", 1.0]]',
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    written = list(records_reader(payload["split_manifest_path"]))
    assert all(record.split == "train" for record in written)


# --- build subcommand ----------------------------------------------------------


def test_build_subcommand_prints_record_with_run_id_and_counts(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    split_manifest = _make_split_manifest(tmp_path, images, tmp_path / "split")

    exit_code = etl_group.run(
        [
            "build",
            "--output=json",
            f"split_manifest={split_manifest}",
            f"shard_root={tmp_path / 'shards'}",
            "workers=1",
            # See the analogous extract-subcommand test for why the runner
            # group is dropped: local.yaml's default task runner needs a
            # live Prefect flow run to map tasks under, unreachable here.
            "~runner",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert set(payload) == {
        "run_id",
        "output_dir",
        "manifest_path",
        "report_path",
        "shard_count",
    }
    assert payload["shard_count"] >= 1
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["report_path"]).exists()


def test_build_subcommand_applies_config_overrides(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=4)
    split_manifest = _make_split_manifest(tmp_path, images, tmp_path / "split")

    exit_code = etl_group.run(
        [
            "build",
            "--output=json",
            f"split_manifest={split_manifest}",
            f"shard_root={tmp_path / 'shards'}",
            "workers=1",
            "shard_size=1",
            "~runner",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    # 8 images all assigned to "train" split -> shard_size=1 -> 8 shards.
    assert payload["shard_count"] == 8


def test_build_runner_override_selects_the_runner_family(
    tmp_path: Path,
    clear_global_hydra: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """See ``test_extract_runner_override_selects_the_runner_family`` for why
    the flow body is a no-op here: a real attached task runner needs
    mapped-task execution under a live Prefect engine, unreachable in this
    environment."""
    from radiologist.etl.models import BuildResult

    images = _build_image_tree(tmp_path)
    split_manifest = _make_split_manifest(tmp_path, images, tmp_path / "split")

    real_resolve = prefect_pipelines.resolve_execution
    seen_families: List[str] = []

    def _spy(runner_cfg, batch_size=None):
        plan = real_resolve(runner_cfg, batch_size=batch_size)
        seen_families.append(plan.family)
        return plan

    monkeypatch.setattr(prefect_pipelines, "resolve_execution", _spy)
    fake_result = BuildResult(
        run_id="r",
        output_dir="o",
        manifest_path="m",
        report_path="rp",
        shard_count=0,
        record_count=0,
    )
    monkeypatch.setattr(
        prefect_pipelines,
        "build_flow",
        _FlowSpy(lambda cfg, execution=None: fake_result),
    )

    exit_code = etl_group.run(
        [
            "build",
            f"split_manifest={split_manifest}",
            f"shard_root={tmp_path / 'shards'}",
            "workers=1",
            "runner=local",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    assert exit_code == 0
    assert seen_families == ["local"]


# --- global --output flag position -------------------------------------------


def test_output_flag_honoured_before_subcommand_token(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    manifests_dir = tmp_path / "manifests"
    _make_extract_manifest(tmp_path, images, manifests_dir)

    exit_code = etl_group.run(
        [
            "--output=json",
            "assign-split",
            f"manifests_dir={manifests_dir}",
            f"destination={tmp_path / 'dest'}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert "run_id" in payload


def test_output_flag_honoured_after_subcommand_token(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)
    manifests_dir = tmp_path / "manifests"
    _make_extract_manifest(tmp_path, images, manifests_dir)

    exit_code = etl_group.run(
        [
            "assign-split",
            "--output=json",
            f"manifests_dir={manifests_dir}",
            f"destination={tmp_path / 'dest'}",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert "run_id" in payload


# --- storage_options resolution -------------------------------------------------


def test_cli_storage_options_resolution_treats_explicit_empty_dict_as_valid() -> None:
    """The CLI's pre-flight ``storage_options`` resolution is the same shared
    ``radiologist.etl.storage_options_from_cfg`` the ETL stages use (issue
    #191 review finding 4) — an explicitly configured empty dict must not
    collapse to ``None``, since ``None`` and ``{}`` are different configs."""
    from omegaconf import OmegaConf

    from radiologist.etl import storage_options_from_cfg

    cfg = OmegaConf.create({"storage_options": {}})

    assert etl_group._storage_options_from_cfg(cfg) == {}
    assert etl_group._storage_options_from_cfg is storage_options_from_cfg


# --- package export cutover ---------------------------------------------------


def test_etl_package_no_longer_exposes_the_monolithic_flow_surface() -> None:
    import radiologist.etl as etl_pkg

    removed_names = {
        "etl_flow",
        "EtlResult",
        "StatsProcessor",
        "compute_run_id",
        "apply_filters_task",
        "assign_splits_task",
        "build_shards_task",
        "compute_stats_task",
        "write_jsonl_task",
    }
    for name in removed_names:
        assert not hasattr(etl_pkg, name), f"{name} should no longer be exported"
        assert name not in etl_pkg.__all__
