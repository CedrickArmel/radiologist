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

"""Behavioral tests for ``extract_flow``/``assign_split_flow``/``build_flow``
and their ``run_extract``/``run_assign_split``/``run_build`` handoff wiring.

The monolithic ``etl_flow``/``EtlResult`` surface these tests used to cover
was retired by the CLI cutover issue (#187): the three per-stage flows below
re-establish every behavior it asserted, so its own tests were deleted here
rather than migrated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import radiologist.etl.prefect_pipelines as prefect_pipelines
from radiologist.etl.assign import assign_splits  # noqa: E402
from radiologist.etl.build import build_shards as build_shards_stage  # noqa: E402
from radiologist.etl.execution import ExecutionPlan  # noqa: E402
from radiologist.etl.extract import extract  # noqa: E402
from radiologist.etl.stats import lung_asymmetry, make_haralick  # noqa: E402

# --- issue #186: extract_flow / assign_split_flow / build_flow -------------
#
# Real local Prefect flow execution (any call that is not the ``.fn`` escape
# hatch) needs a live Prefect API and this environment's installed Prefect
# 3.7.4 cannot reach one — confirmed independently against both the
# ephemeral local server and a manually started ``prefect server start``
# instance; both fail with the same third-party Starlette/FastAPI routing
# mismatch (``AttributeError: 'PrefectRouter' object has no attribute
# 'routes'``), unrelated to this change. See
# feedback_prefect_broken_local_server_use_fn_bypass memory. Every test
# below therefore drives real ``radiologist.etl`` business logic via the
# same ``.fn``/artifact-stub pattern already used above for ``etl_flow``,
# and — where a test needs to exercise this issue's own new wiring code
# (``with_task_runner``, the plan's task-runner-vs-beam mapper dispatch,
# ``run_extract``/``run_build``'s flow handoff) without invoking the broken
# engine — patches only the third-party ``Flow``/``Task`` call boundary
# (``.map()``, or the module-level flow object itself), never any
# `radiologist.etl` business logic.


def _extract_cfg(file_list: Path, destination: Path, **overrides: object) -> object:
    base: dict[str, object] = {
        "file_list": str(file_list),
        "destination": str(destination),
        "images_root": None,
        "masks_root": None,
        "iqr_columns": [],
        "iqr_factor": 1.5,
        "haralick": {"features": ["contrast"], "distances": None, "angles": None},
        "workers": 1,
        "batch_size": 64,
        "max_failure_rate": 0.0,
        "run_label": None,
        "storage_options": None,
        "runner": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _build_cfg(split_manifest: Path, shard_root: Path, **overrides: object) -> object:
    base: dict[str, object] = {
        "split_manifest": str(split_manifest),
        "shard_root": str(shard_root),
        "shard_size": 1000,
        "split_ratios": [["train", 0.70], ["val", 0.15], ["test", 0.15]],
        "workers": 1,
        "run_label": None,
        "storage_options": None,
        "runner": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _assign_split_cfg(
    manifests_dir: Path, destination: Path, **overrides: object
) -> object:
    base: dict[str, object] = {
        "manifests_dir": str(manifests_dir),
        "destination": str(destination),
        "split_ratios": [["train", 0.70], ["val", 0.15], ["test", 0.15]],
        "run_label": None,
        "storage_options": None,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _write_listing(tmp_path: Path, paths: list, name: str = "listing.txt") -> str:
    listing_path = tmp_path / name
    listing_path.write_text("\n".join(paths) + "\n")
    return str(listing_path)


def _all_image_paths(image_dir: Path) -> list:
    return [str(p) for p in sorted(image_dir.rglob("*.png"))]


@pytest.fixture(autouse=True)
def _pop_prefect_cloud_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PREFECT_API_URL", raising=False)
    monkeypatch.delenv("PREFECT_API_KEY", raising=False)


class _FakeBeamExecutor:
    """Stand-in exposing the two mapper methods a real BeamExecutor will,
    per the issue's own instruction: the Beam-shaped dispatch branch must be
    verified with a stand-in object, not the real (still-stubbed)
    BeamExecutor, so this test never needs to change once #189 lands."""

    def __init__(self) -> None:
        self.batch_calls: list = []
        self.shard_calls: list = []

    def run_batches(self, batches, images_root, masks_root, manifest_id, extractors):
        self.batch_calls.append(batches)
        from radiologist.etl.processors import process_batch

        return [
            process_batch(
                batch,
                images_root=images_root,
                masks_root=masks_root,
                manifest_id=manifest_id,
                extractors=extractors,
            )
            for batch in batches
        ]

    def run_shards(self, jobs):
        self.shard_calls.append(jobs)
        from radiologist.etl.shards import write_shard

        return [write_shard(job) for job in jobs]


# --- with_task_runner --------------------------------------------------------


def test_with_task_runner_attaches_the_plans_task_runner_when_present() -> None:
    from prefect.task_runners import ProcessPoolTaskRunner

    plan = ExecutionPlan(family="local", task_runner=ProcessPoolTaskRunner())

    result = prefect_pipelines.with_task_runner(prefect_pipelines.extract_flow, plan)

    assert result.task_runner is plan.task_runner


def test_with_task_runner_returns_the_flow_unchanged_when_no_task_runner() -> None:
    plan = ExecutionPlan(family="local", task_runner=None)

    result = prefect_pipelines.with_task_runner(prefect_pipelines.extract_flow, plan)

    assert result is prefect_pipelines.extract_flow


# --- extract_batch_task / write_shard_task -----------------------------------


def test_extract_batch_task_delegates_to_process_batch(image_dir: Path) -> None:
    paths = _all_image_paths(image_dir)

    outcome = prefect_pipelines.extract_batch_task.fn(
        paths=paths,
        images_root=str(image_dir),
        masks_root=None,
        manifest_id="run00000000000001",
        extractors=[],
    )

    assert len(outcome.records) == len(paths)
    assert all(r.manifest_id == "run00000000000001" for r in outcome.records)


def test_write_shard_task_delegates_to_write_shard(tmp_path: Path) -> None:
    from radiologist.etl.models import ShardJob

    job = ShardJob(
        split="train", label="NORMAL", index=0, shard_root=str(tmp_path), records=[]
    )

    outcome = prefect_pipelines.write_shard_task.fn(job)

    assert outcome.relative_path == "train/NORMAL/train-normal-000000.tar"
    assert (tmp_path / "train" / "NORMAL" / "train-normal-000000.tar").exists()


# --- extract_flow -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bypass_extract_and_build_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefect_pipelines, "create_link_artifact", lambda **_: None)
    monkeypatch.setattr(prefect_pipelines, "create_markdown_artifact", lambda **_: None)
    monkeypatch.setattr(prefect_pipelines, "create_table_artifact", lambda **_: None)


def test_extract_flow_with_no_runner_matches_calling_extract_directly(
    image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    flow_result = prefect_pipelines.extract_flow.fn(cfg)
    # Same extractor construction extract_flow derives from cfg.haralick, so
    # the direct call is a true apples-to-apples comparison (same run id).
    direct_extractors = [
        make_haralick(features=["contrast"], distances=None, angles=None),
        lung_asymmetry,
    ]
    direct_result = extract(
        listing,
        str(destination / "direct"),
        extractors=direct_extractors,
        iqr_columns=[],
        run_label=None,
    )

    assert flow_result.run_id == direct_result.run_id
    assert (
        Path(flow_result.manifest_path).read_bytes()
        == Path(direct_result.manifest_path).read_bytes()
    )


def test_extract_flow_returns_extract_result_with_run_id_and_manifest_path(
    image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    result = prefect_pipelines.extract_flow.fn(cfg)

    assert Path(result.manifest_path).exists()
    assert result.run_id in result.manifest_path
    assert result.total == len(paths)


def test_extract_flow_links_manifest_artifact_with_run_counts(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    captured: dict = {}

    def _capture_link(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(prefect_pipelines, "create_link_artifact", _capture_link)

    result = prefect_pipelines.extract_flow.fn(cfg)

    assert captured["link"] == result.manifest_path
    assert str(result.total) in captured["description"]
    assert str(result.succeeded) in captured["description"]
    assert str(result.failed) in captured["description"]
    assert str(result.excluded) in captured["description"]


def test_extract_flow_records_config_artifact_naming_the_runner_family(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    captured: dict = {}

    def _capture_markdown(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        prefect_pipelines, "create_markdown_artifact", _capture_markdown
    )

    prefect_pipelines.extract_flow.fn(cfg)

    assert "local" in captured["markdown"]


def test_extract_flow_warns_when_prefect_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    image_dir: Path,
    tmp_path: Path,
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    monkeypatch.setattr(prefect_pipelines, "_PREFECT_AVAILABLE", False)

    with caplog.at_level("WARNING"):
        result = prefect_pipelines.extract_flow.fn(cfg)

    assert Path(result.manifest_path).exists()
    assert any(
        "not being recorded" in m or "prefect" in m.lower() for m in caplog.messages
    )


def test_extract_flow_with_beam_shaped_plan_uses_the_executors_run_batches_as_mapper(
    image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    fake_beam = _FakeBeamExecutor()
    plan = ExecutionPlan(family="beam", task_runner=None, beam=fake_beam)

    result = prefect_pipelines.extract_flow.fn(cfg, execution=plan)

    assert fake_beam.batch_calls, "expected the beam executor's run_batches to be used"
    assert Path(result.manifest_path).exists()
    assert result.total == len(paths)


# --- build_flow ----------------------------------------------------------------


def _make_split_manifest(tmp_path: Path, image_dir: Path) -> str:
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
    manifest_path = str(
        tmp_path / "split-manifests" / "manifest-assignrun0000001.jsonl"
    )
    JsonlWriter().write(records, manifest_path)
    return manifest_path


def test_build_flow_with_no_runner_matches_calling_build_shards_directly(
    image_dir: Path, tmp_path: Path
) -> None:
    split_manifest = _make_split_manifest(tmp_path, image_dir)
    shard_root = tmp_path / "shards"
    cfg = _build_cfg(Path(split_manifest), shard_root)

    flow_result = prefect_pipelines.build_flow.fn(cfg)
    direct_result = build_shards_stage(
        split_manifest,
        str(shard_root),
        ratios=(("train", 0.70), ("val", 0.15), ("test", 0.15)),
    )

    assert flow_result.run_id == direct_result.run_id
    assert flow_result.shard_count == direct_result.shard_count
    assert (
        Path(flow_result.manifest_path).read_bytes()
        == Path(direct_result.manifest_path).read_bytes()
    )


def test_build_flow_links_output_artifact_with_shard_and_record_counts(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    split_manifest = _make_split_manifest(tmp_path, image_dir)
    shard_root = tmp_path / "shards"
    cfg = _build_cfg(Path(split_manifest), shard_root)

    captured: dict = {}
    monkeypatch.setattr(
        prefect_pipelines, "create_link_artifact", lambda **kw: captured.update(kw)
    )

    result = prefect_pipelines.build_flow.fn(cfg)

    assert captured["link"] == result.output_dir
    assert str(result.shard_count) in captured["description"]
    assert str(result.record_count) in captured["description"]


def test_build_flow_reports_the_split_report_as_a_table_artifact(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    split_manifest = _make_split_manifest(tmp_path, image_dir)
    shard_root = tmp_path / "shards"
    cfg = _build_cfg(Path(split_manifest), shard_root)

    captured: dict = {}
    monkeypatch.setattr(
        prefect_pipelines, "create_table_artifact", lambda **kw: captured.update(kw)
    )

    prefect_pipelines.build_flow.fn(cfg)

    assert captured["table"]
    assert "label" in captured["table"][0]


def test_build_flow_with_beam_shaped_plan_uses_the_executors_run_shards_as_mapper(
    image_dir: Path, tmp_path: Path
) -> None:
    split_manifest = _make_split_manifest(tmp_path, image_dir)
    shard_root = tmp_path / "shards"
    cfg = _build_cfg(Path(split_manifest), shard_root)

    fake_beam = _FakeBeamExecutor()
    plan = ExecutionPlan(family="beam", task_runner=None, beam=fake_beam)

    result = prefect_pipelines.build_flow.fn(cfg, execution=plan)

    assert fake_beam.shard_calls, "expected the beam executor's run_shards to be used"
    assert result.shard_count > 0


# --- assign_split_flow -----------------------------------------------------------


def test_assign_split_flow_matches_calling_assign_splits_directly(
    tmp_path: Path,
) -> None:
    from radiologist.etl.manifest import JsonlWriter, ManifestRecord

    manifests_dir = tmp_path / "manifests"
    record = ManifestRecord(
        manifest_id="extractrun000001",
        path="/img/a.png",
        filename="a.png",
        label="NORMAL",
        split="",
        stats={},
    )
    JsonlWriter().write([record], str(manifests_dir / "extract-extractrun000001.jsonl"))
    destination = tmp_path / "dest"
    cfg = _assign_split_cfg(manifests_dir, destination)

    flow_result = prefect_pipelines.assign_split_flow.fn(cfg)
    direct_result = assign_splits(
        str(manifests_dir),
        str(destination),
        ratios=(("train", 0.70), ("val", 0.15), ("test", 0.15)),
    )

    assert flow_result.run_id == direct_result.run_id
    assert (
        Path(flow_result.split_manifest_path).read_bytes()
        == Path(direct_result.split_manifest_path).read_bytes()
    )


def test_assign_split_flow_links_split_manifest_artifact_with_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from radiologist.etl.manifest import JsonlWriter, ManifestRecord

    manifests_dir = tmp_path / "manifests"
    record = ManifestRecord(
        manifest_id="extractrun000001",
        path="/img/a.png",
        filename="a.png",
        label="NORMAL",
        split="",
        stats={},
    )
    JsonlWriter().write([record], str(manifests_dir / "extract-extractrun000001.jsonl"))
    destination = tmp_path / "dest"
    cfg = _assign_split_cfg(manifests_dir, destination)

    captured: dict = {}
    monkeypatch.setattr(
        prefect_pipelines, "create_link_artifact", lambda **kw: captured.update(kw)
    )

    result = prefect_pipelines.assign_split_flow.fn(cfg)

    assert captured["link"] == result.split_manifest_path
    assert str(result.source_manifest_count) in captured["description"]
    assert str(result.duplicate_count) in captured["description"]


# --- run_extract / run_assign_split / run_build (flow-handoff wiring) --------


class _FakeFlow:
    """Stand-in for a Prefect ``Flow`` object: mimics ``.with_options()`` and
    a direct call, without needing the (broken-in-this-environment) live
    Prefect engine. Used only to verify this issue's own new wiring code —
    that ``run_extract``/``run_build`` resolve a plan, attach its task
    runner, and forward ``cfg``/``execution`` correctly — never to replace
    any ``radiologist.etl`` business logic."""

    def __init__(self, fn):
        self.fn = fn
        self.task_runner = None
        self.calls: list = []
        self.with_options_calls: list = []

    def with_options(self, task_runner=None):
        self.with_options_calls.append({"task_runner": task_runner})
        new = _FakeFlow(self.fn)
        new.task_runner = task_runner
        new.calls = self.calls
        new.with_options_calls = self.with_options_calls
        return new

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


def test_run_extract_resolves_the_plan_and_attaches_its_task_runner(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    from prefect.task_runners import ProcessPoolTaskRunner

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(
        Path(listing),
        destination,
        runner={
            "family": "local",
            "batch_size": 64,
            "task_runner": {
                "_target_": "prefect.task_runners.ProcessPoolTaskRunner",
                "max_workers": None,
            },
        },
    )

    # A resolved task runner makes the real flow body build a mapped-task
    # mapper, which needs a live flow run (Prefect's engine — broken in this
    # environment, see the module-level note above). This test only proves
    # run_extract's own wiring — that it resolves the plan and attaches its
    # task runner before invoking the flow — so the patched flow body is a
    # no-op that never touches ``.map()``.
    fake = _FakeFlow(lambda cfg_arg, execution=None: "not-a-real-run")
    monkeypatch.setattr(prefect_pipelines, "extract_flow", fake)

    prefect_pipelines.run_extract(cfg)

    assert fake.with_options_calls, "expected run_extract to attach a task runner"
    assert isinstance(fake.with_options_calls[0]["task_runner"], ProcessPoolTaskRunner)


def test_run_extract_passes_execution_plan_through_to_the_flow_call(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = tmp_path / "dest"
    cfg = _extract_cfg(Path(listing), destination)

    real_fn = prefect_pipelines.extract_flow.fn
    seen_execution: list = []

    def _fake_fn(cfg_arg, execution=None):
        seen_execution.append(execution)
        return real_fn(cfg_arg, execution=execution)

    fake = _FakeFlow(_fake_fn)
    monkeypatch.setattr(prefect_pipelines, "extract_flow", fake)

    prefect_pipelines.run_extract(cfg)

    assert seen_execution and isinstance(seen_execution[0], ExecutionPlan)
    assert seen_execution[0].family == "local"


def test_run_assign_split_calls_assign_split_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from radiologist.etl.manifest import JsonlWriter, ManifestRecord

    manifests_dir = tmp_path / "manifests"
    record = ManifestRecord(
        manifest_id="extractrun000001",
        path="/img/a.png",
        filename="a.png",
        label="NORMAL",
        split="",
        stats={},
    )
    JsonlWriter().write([record], str(manifests_dir / "extract-extractrun000001.jsonl"))
    destination = tmp_path / "dest"
    cfg = _assign_split_cfg(manifests_dir, destination)

    fake = _FakeFlow(prefect_pipelines.assign_split_flow.fn)
    monkeypatch.setattr(prefect_pipelines, "assign_split_flow", fake)

    result = prefect_pipelines.run_assign_split(cfg)

    assert Path(result.split_manifest_path).exists()
    assert fake.calls


def test_run_build_resolves_the_plan_and_attaches_its_task_runner(
    monkeypatch: pytest.MonkeyPatch, image_dir: Path, tmp_path: Path
) -> None:
    split_manifest = _make_split_manifest(tmp_path, image_dir)
    shard_root = tmp_path / "shards"
    cfg = _build_cfg(Path(split_manifest), shard_root)

    real_fn = prefect_pipelines.build_flow.fn
    seen_execution: list = []

    def _fake_fn(cfg_arg, execution=None):
        seen_execution.append(execution)
        return real_fn(cfg_arg, execution=execution)

    fake = _FakeFlow(_fake_fn)
    monkeypatch.setattr(prefect_pipelines, "build_flow", fake)

    result = prefect_pipelines.run_build(cfg)

    assert Path(result.manifest_path).exists()
    assert seen_execution and isinstance(seen_execution[0], ExecutionPlan)
