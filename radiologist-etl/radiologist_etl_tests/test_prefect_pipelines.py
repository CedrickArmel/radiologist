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

"""Behavioral tests for ``etl_flow``'s ``EtlResult`` return contract.

These tests call ``etl_flow.fn(cfg)`` — Prefect's own documented escape
hatch for invoking a ``@flow``-wrapped function's underlying callable
directly — and likewise call each Prefect ``@task`` through its own
``.fn`` (also Prefect's own documented escape hatch), plus stub the
module's artifact-creation calls (a true HTTP boundary to the Prefect
API). None of this skips any ETL business logic — every stage still runs
for real; it only bypasses Prefect's orchestration/tracking layer, which
this environment's installed Prefect 3.7.4 cannot reach: it is
incompatible with the resolved Starlette release (a genuine third-party
version mismatch, unrelated to this change) and its ephemeral API server
fails under pytest. This is the same reason ``test_pipelines.py`` stays
``--ignore``d in the root ``pyproject.toml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

import radiologist.etl.prefect_pipelines as prefect_pipelines
from radiologist.etl import EtlResult, compute_run_id, etl_flow


@pytest.fixture(autouse=True)
def _bypass_prefect_orchestration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route calls through Prefect's ``.fn`` escape hatch and stub its
    artifact HTTP calls, so real ETL logic runs without needing a live
    Prefect API server (unreachable in this environment; see module
    docstring)."""
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


def _minimal_cfg(
    images_root: Path,
    destination: Path,
    artifact_dir: Path,
    run_label: str,
) -> object:
    return OmegaConf.create(
        {
            "source": str(images_root),
            "masks_root": None,
            "destination": str(destination),
            "artifact_dir": str(artifact_dir),
            "iqr_columns": [],
            "iqr_factor": 1.5,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "workers": 1,
            "storage_options": None,
            "build_shards": False,
            "shard_root": None,
            "shard_size": 1000,
            "run_label": run_label,
            "resume_from_parquet": None,
            "resume_from_filtered": None,
            "resume_from_split": None,
            "resume_from_manifest": None,
            "haralick": {"features": ["contrast"], "distances": None, "angles": None},
        }
    )


def test_etl_flow_returns_etl_result_with_run_id_and_manifest_path(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", "r1")

    result = etl_flow.fn(cfg)

    assert isinstance(result, EtlResult)
    assert Path(result.manifest_path).exists()
    with open(result.manifest_path, "rt", encoding="utf-8") as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    assert len(records) == 4


def test_etl_flow_run_id_matches_the_run_id_used_to_name_artifacts(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", "r2")

    result = etl_flow.fn(cfg)

    expected_run_id = compute_run_id(cfg, str(images))
    assert result.run_id == expected_run_id
    assert result.run_id in result.manifest_path
