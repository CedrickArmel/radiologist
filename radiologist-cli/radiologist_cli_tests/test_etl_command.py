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

"""Behavioral tests for the ``radiologist etl`` command group.

``--help``, ``--cfg job`` (override reflection) and the source-not-found
path are exercised via ``subprocess`` (mirroring the deleted
``radiologist-etl/radiologist_etl_tests/test_console_script.py``) since
they don't require running the real pipeline.

The multirun, ``--output=json`` and "other pipeline failure" scenarios run
the real ETL business logic in-process with Prefect's own
orchestration/tracking HTTP calls stubbed out (a true process boundary —
see ``radiologist-etl/radiologist_etl_tests/test_prefect_pipelines.py``
for the same technique and its rationale): this environment has no
reachable Prefect API (the local ephemeral server is broken by a
Starlette/Prefect version mismatch unrelated to this change, and the
configured Prefect Cloud credentials must not be hit by tests).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image

import radiologist.cli.groups.etl as etl_group
import radiologist.etl.prefect_pipelines as prefect_pipelines

CLI_ROOT = Path(__file__).resolve().parents[1]


def _run_cli_subprocess(argv: List[str]) -> Tuple[int, str, str]:
    env = dict(os.environ)
    env.pop("PREFECT_API_URL", None)
    env.pop("PREFECT_API_KEY", None)
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


@pytest.fixture()
def bypass_prefect_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Prefect's HTTP orchestration/artifact calls (a true process
    boundary), leaving all real ETL business logic untouched."""
    monkeypatch.setattr(prefect_pipelines, "create_link_artifact", lambda **_: None)
    monkeypatch.setattr(prefect_pipelines, "create_markdown_artifact", lambda **_: None)
    monkeypatch.setattr(prefect_pipelines, "create_table_artifact", lambda **_: None)
    for task_name in (
        "compute_stats_task",
        "apply_filters_task",
        "assign_splits_task",
        "write_jsonl_task",
        "build_shards_task",
    ):
        task_obj = getattr(prefect_pipelines, task_name)
        monkeypatch.setattr(prefect_pipelines, task_name, task_obj.fn)
    monkeypatch.setattr(etl_group, "etl_flow", prefect_pipelines.etl_flow.fn)


@pytest.fixture()
def clear_global_hydra():
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()


def test_help_exits_zero_and_prints_composed_config_tree() -> None:
    returncode, stdout, _ = _run_cli_subprocess(["--help"])

    assert returncode == 0
    assert "Powered by Hydra" in stdout


def test_key_value_override_is_reflected_in_composed_config() -> None:
    returncode, stdout, _ = _run_cli_subprocess(["--cfg", "job", "iqr_factor=42.0"])

    assert returncode == 0
    assert "iqr_factor: 42.0" in stdout


def test_source_that_does_not_exist_exits_2(tmp_path: Path) -> None:
    missing_source = tmp_path / "does-not-exist"
    returncode, _, stderr = _run_cli_subprocess(
        [
            f"source={missing_source}",
            f"destination={tmp_path / 'out'}",
            f"artifact_dir={tmp_path / 'artifacts'}",
        ]
    )

    assert returncode == 2
    assert "Error: " in stderr


def test_pipeline_failure_for_other_reason_exits_1_with_error_on_stderr(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    empty_source = tmp_path / "empty_source"
    empty_source.mkdir()

    exit_code = etl_group.run(
        [
            f"source={empty_source}",
            f"destination={tmp_path / 'out'}",
            f"artifact_dir={tmp_path / 'artifacts'}",
            "run_label=fail1",
            "masks_root=null",
            "iqr_columns=[]",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error: " in captured.err


def test_output_json_produces_one_parseable_object_with_both_keys(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
    capsys: pytest.CaptureFixture,
) -> None:
    images = _build_image_tree(tmp_path)

    exit_code = etl_group.run(
        [
            "--output=json",
            f"source={images}",
            f"destination={tmp_path / 'out'}",
            f"artifact_dir={tmp_path / 'artifacts'}",
            "run_label=json1",
            "workers=1",
            "build_shards=false",
            "masks_root=null",
            "iqr_columns=[]",
            f"hydra.run.dir={tmp_path / 'hydra_run'}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert set(payload) == {"run_id", "manifest_path"}


def test_multirun_sweep_launches_one_job_per_swept_value(
    tmp_path: Path,
    bypass_prefect_orchestration: None,
    clear_global_hydra: None,
) -> None:
    images = _build_image_tree(tmp_path)

    exit_code = etl_group.run(
        [
            "--multirun",
            f"source={images}",
            f"destination={tmp_path / 'out'}",
            f"artifact_dir={tmp_path / 'artifacts'}",
            "run_label=sweep_a,sweep_b",
            "workers=1",
            "build_shards=false",
            "masks_root=null",
            "iqr_columns=[]",
            f"hydra.sweep.dir={tmp_path / 'multirun'}",
        ]
    )

    assert exit_code == 0
    manifests = sorted((tmp_path / "out").glob("manifest-*.jsonl"))
    assert len(manifests) == 2
