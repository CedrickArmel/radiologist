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

"""Behavioral tests for the Apache Beam execution family (#189).

Verification uses Beam's direct runner only: it is the one runner that can
be exercised in-process. Managed and cluster runners are covered by shape —
their options mapping must reach the executor verbatim, and construction
must reject a parts location the workers could not reach.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

import radiologist.etl.prefect_pipelines as prefect_pipelines
from radiologist.etl import (
    BeamExecutor,
    ExecutionPlan,
    ExtractionFailureError,
    JsonlWriter,
    ManifestRecord,
    plan_shards,
    resolve_execution,
)

# --- helpers -----------------------------------------------------------------


def _compose(config_name: str, overrides: list | None = None):
    with initialize_config_module(
        config_module="radiologist.etl.conf", version_base=None
    ):
        return compose(config_name=config_name, overrides=overrides or [])


def _direct_executor(parts_dir: Path, **options: object) -> BeamExecutor:
    pipeline_options: dict = {"runner": "DirectRunner"}
    pipeline_options.update(options)
    return BeamExecutor(pipeline_options=pipeline_options, parts_dir=str(parts_dir))


def _extract_cfg(file_list: str, destination: Path, **overrides: object) -> object:
    base: dict[str, object] = {
        "file_list": file_list,
        "destination": str(destination),
        "images_root": None,
        "masks_root": None,
        "iqr_columns": [],
        "iqr_factor": 1.5,
        "haralick": {"features": ["contrast"], "distances": None, "angles": None},
        "workers": 1,
        "batch_size": 2,
        "max_failure_rate": 0.0,
        "run_label": None,
        "storage_options": None,
        "runner": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _build_cfg(split_manifest: str, shard_root: Path, **overrides: object) -> object:
    base: dict[str, object] = {
        "split_manifest": split_manifest,
        "shard_root": str(shard_root),
        "shard_size": 1,
        "split_ratios": [["train", 0.70], ["val", 0.15], ["test", 0.15]],
        "workers": 1,
        "run_label": None,
        "storage_options": None,
        "runner": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _write_listing(tmp_path: Path, paths: list, name: str = "listing.txt") -> str:
    listing_path = tmp_path / name
    listing_path.write_text("\n".join(paths) + "\n")
    return str(listing_path)


def _all_image_paths(image_dir: Path) -> list:
    return [str(p) for p in sorted(image_dir.rglob("*.png"))]


def _split_records(image_dir: Path) -> list:
    return [
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


def _make_split_manifest(tmp_path: Path, image_dir: Path) -> str:
    manifest_path = str(
        tmp_path / "split-manifests" / "manifest-assignrun0000001.jsonl"
    )
    JsonlWriter().write(_split_records(image_dir), manifest_path)
    return manifest_path


def _shard_contents(output_dir: str) -> dict:
    """Map every tar shard's relative name to its sorted (member, bytes) payload."""
    root = Path(output_dir)
    contents: dict = {}
    for tar_path in sorted(root.rglob("*.tar")):
        with tarfile.open(tar_path) as tar:
            members = sorted(
                (m.name, tar.extractfile(m).read())  # type: ignore[union-attr]
                for m in tar.getmembers()
                if m.isfile()
            )
        contents[str(tar_path.relative_to(root))] = members
    return contents


class _CountingBeamExecutor(BeamExecutor):
    """A real executor that also records how many times it was dispatched to."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.run_batches_calls = 0

    def run_batches(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.run_batches_calls += 1
        return super().run_batches(*args, **kwargs)


# --- runner selection --------------------------------------------------------


def test_direct_beam_choice_yields_a_beam_plan_with_an_executor_and_no_task_runner(
    tmp_path: Path,
) -> None:
    cfg = _compose(
        "extract",
        overrides=[
            "runner=beam_direct",
            f"runner.beam.parts_dir={tmp_path / 'parts'}",
        ],
    )

    plan = resolve_execution(cfg.runner)

    assert plan.family == "beam"
    assert plan.task_runner is None
    assert isinstance(plan.beam, BeamExecutor)


def test_the_shipped_direct_beam_config_runs_a_pipeline_end_to_end(
    image_dir: Path, tmp_path: Path
) -> None:
    """The composed ``runner=beam_direct`` node — not a hand-built executor —
    must be a working pipeline configuration as shipped."""
    cfg = _compose(
        "extract",
        overrides=[
            "runner=beam_direct",
            f"runner.beam.parts_dir={tmp_path / 'parts'}",
        ],
    )
    executor = resolve_execution(cfg.runner).beam
    assert executor is not None
    paths = _all_image_paths(image_dir)

    outcomes = executor.run_batches(
        [[p] for p in paths],
        images_root=None,
        masks_root=None,
        manifest_id="beamrun000000001",
        extractors=[],
    )

    assert [[r.path for r in o.records] for o in outcomes] == [[p] for p in paths]


def test_dataflow_choice_carries_its_pipeline_options_through_verbatim() -> None:
    cfg = _compose(
        "extract",
        overrides=[
            "runner=beam_dataflow",
            "runner.beam.parts_dir=gs://bucket/parts",
            "runner.beam.pipeline_options.project=a-project",
            "runner.beam.pipeline_options.region=europe-west1",
            "runner.beam.pipeline_options.temp_location=gs://bucket/tmp",
            "runner.beam.pipeline_options.staging_location=gs://bucket/staging",
        ],
    )

    plan = resolve_execution(cfg.runner)

    assert plan.beam is not None
    assert plan.beam.pipeline_options == {
        "runner": "DataflowRunner",
        "project": "a-project",
        "region": "europe-west1",
        "temp_location": "gs://bucket/tmp",
        "staging_location": "gs://bucket/staging",
        "sdk_container_image": None,
        "max_num_workers": None,
    }
    assert plan.beam.parts_dir == "gs://bucket/parts"


def test_a_non_direct_runner_with_a_local_parts_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reachable"):
        BeamExecutor(
            pipeline_options={"runner": "DataflowRunner"},
            parts_dir=str(tmp_path / "parts"),
        )


def test_selecting_beam_without_the_backend_names_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from radiologist.etl import optional

    monkeypatch.setattr(optional, "_BEAM_AVAILABLE", False)

    with pytest.raises(RuntimeError, match=r"radiologist-etl\[beam\]"):
        BeamExecutor(
            pipeline_options={"runner": "DirectRunner"},
            parts_dir=str(tmp_path / "parts"),
        )


# --- run_batches / run_shards ------------------------------------------------


def test_run_batches_returns_one_outcome_per_batch_in_input_order(
    image_dir: Path, tmp_path: Path
) -> None:
    executor = _direct_executor(tmp_path / "parts")
    paths = _all_image_paths(image_dir)
    batches = [[p] for p in paths]

    outcomes = executor.run_batches(
        batches,
        images_root=None,
        masks_root=None,
        manifest_id="beamrun000000001",
        extractors=[],
    )

    assert [[r.path for r in o.records] for o in outcomes] == batches
    assert all(r.manifest_id == "beamrun000000001" for o in outcomes for r in o.records)


def test_run_shards_returns_one_outcome_per_job_in_input_order(
    image_dir: Path, tmp_path: Path
) -> None:
    executor = _direct_executor(tmp_path / "parts")
    jobs = plan_shards(
        _split_records(image_dir), str(tmp_path / "shards"), shard_size=1
    )

    outcomes = executor.run_shards(jobs)

    assert len(outcomes) == len(jobs)
    assert [o.relative_path for o in outcomes] == [
        f"{job.split}/{job.label}/{job.split}-{job.label.lower()}-{job.index:06d}.tar"
        for job in jobs
    ]
    assert all(o.written == 1 for o in outcomes)


def test_parts_from_an_earlier_run_do_not_affect_a_later_run_over_other_inputs(
    image_dir: Path, tmp_path: Path
) -> None:
    executor = _direct_executor(tmp_path / "parts")
    paths = _all_image_paths(image_dir)

    executor.run_batches(
        [paths[:2], paths[2:]],
        images_root=None,
        masks_root=None,
        manifest_id="first-run-00000",
        extractors=[],
    )
    second = executor.run_batches(
        [paths[:1]],
        images_root=None,
        masks_root=None,
        manifest_id="second-run-0000",
        extractors=[],
    )

    assert len(second) == 1
    assert [r.path for r in second[0].records] == paths[:1]
    assert all(r.manifest_id == "second-run-0000" for r in second[0].records)


def test_a_failing_beam_pipeline_raises_naming_its_failed_state(
    image_dir: Path, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a regular file, so no part can ever be written beneath it")
    executor = _direct_executor(blocker)

    with pytest.raises(RuntimeError, match="FAILED"):
        executor.run_batches(
            [[p] for p in _all_image_paths(image_dir)],
            images_root=None,
            masks_root=None,
            manifest_id="beamrun000000001",
            extractors=[],
        )


# --- stage parity with a local run -------------------------------------------


def test_extract_with_the_direct_beam_runner_matches_a_local_run(
    image_dir: Path, tmp_path: Path
) -> None:
    listing = _write_listing(tmp_path, _all_image_paths(image_dir))
    plan = ExecutionPlan(
        family="beam", beam=_direct_executor(tmp_path / "parts"), batch_size=2
    )

    beam_result = prefect_pipelines.extract_flow.fn(
        _extract_cfg(listing, tmp_path / "beam-dest"), execution=plan
    )
    local_result = prefect_pipelines.extract_flow.fn(
        _extract_cfg(listing, tmp_path / "local-dest")
    )

    assert beam_result.run_id == local_result.run_id
    assert (
        Path(beam_result.manifest_path).read_bytes()
        == Path(local_result.manifest_path).read_bytes()
    )


def test_build_with_the_direct_beam_runner_matches_a_local_run(
    image_dir: Path, tmp_path: Path
) -> None:
    split_manifest = _make_split_manifest(tmp_path, image_dir)
    plan = ExecutionPlan(family="beam", beam=_direct_executor(tmp_path / "parts"))

    beam_result = prefect_pipelines.build_flow.fn(
        _build_cfg(split_manifest, tmp_path / "beam-shards"), execution=plan
    )
    local_result = prefect_pipelines.build_flow.fn(
        _build_cfg(split_manifest, tmp_path / "local-shards")
    )

    assert beam_result.run_id == local_result.run_id
    assert beam_result.shard_count == local_result.shard_count
    assert _shard_contents(beam_result.output_dir) == _shard_contents(
        local_result.output_dir
    )
    assert (
        Path(beam_result.manifest_path).read_bytes()
        == Path(local_result.manifest_path).read_bytes()
    )


def test_extract_with_beam_surfaces_per_image_failures_like_a_local_run(
    image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir) + [str(tmp_path / "does-not-exist.png")]
    listing = _write_listing(tmp_path, paths)
    plan = ExecutionPlan(
        family="beam", beam=_direct_executor(tmp_path / "parts"), batch_size=2
    )

    with pytest.raises(ExtractionFailureError, match="does-not-exist.png"):
        prefect_pipelines.extract_flow.fn(
            _extract_cfg(listing, tmp_path / "strict-dest"), execution=plan
        )

    tolerant = prefect_pipelines.extract_flow.fn(
        _extract_cfg(listing, tmp_path / "tolerant-dest", max_failure_rate=0.5),
        execution=plan,
    )

    assert tolerant.total == len(paths)
    assert tolerant.failed == 1
    assert tolerant.succeeded == len(paths) - 1


def test_a_beam_backed_extract_is_dispatched_as_a_single_unit_of_work(
    image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    executor = _CountingBeamExecutor(
        pipeline_options={"runner": "DirectRunner"},
        parts_dir=str(tmp_path / "parts"),
    )
    plan = ExecutionPlan(family="beam", beam=executor, batch_size=1)

    result = prefect_pipelines.extract_flow.fn(
        _extract_cfg(listing, tmp_path / "dest", batch_size=1), execution=plan
    )

    assert result.succeeded == len(paths)
    assert executor.run_batches_calls == 1
