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

"""Behavioral tests for the Prefect-native runner-selection surface (#182)."""

from __future__ import annotations

import sys
import types

import pytest
from hydra import compose, initialize_config_module
from hydra.errors import ConfigCompositionException

RUNNER_CHOICES = [
    "local",
    "dask_local",
    "dask_address",
    "dask_cluster",
    "ray_local",
    "ray_cluster",
    "beam_direct",
    "beam_dataflow",
]


def _compose(config_name: str, overrides: list | None = None):
    with initialize_config_module(
        config_module="radiologist.etl.conf", version_base=None
    ):
        return compose(config_name=config_name, overrides=overrides or [])


def test_no_runner_override_yields_local_process_pool_plan():
    from prefect.task_runners import ProcessPoolTaskRunner

    from radiologist.etl.execution import resolve_execution

    cfg = _compose("extract")
    plan = resolve_execution(cfg.runner)

    assert plan.family == "local"
    assert isinstance(plan.task_runner, ProcessPoolTaskRunner)


def test_dask_local_runner_choice_yields_dask_plan_with_worker_count(monkeypatch):
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    fake_module = types.ModuleType("prefect_dask")

    class DaskTaskRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module.DaskTaskRunner = DaskTaskRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefect_dask", fake_module)
    monkeypatch.setattr(optional, "_PREFECT_DASK_AVAILABLE", True)

    cfg = _compose("extract", overrides=["runner=dask_local"])
    plan = resolve_execution(cfg.runner)

    assert plan.family == "dask"
    assert isinstance(plan.task_runner, DaskTaskRunner)
    assert plan.task_runner.kwargs["cluster_kwargs"]["n_workers"] == 4


def test_dask_address_runner_choice_targets_address_without_cluster(monkeypatch):
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    fake_module = types.ModuleType("prefect_dask")

    class DaskTaskRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module.DaskTaskRunner = DaskTaskRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefect_dask", fake_module)
    monkeypatch.setattr(optional, "_PREFECT_DASK_AVAILABLE", True)

    cfg = _compose(
        "extract",
        overrides=[
            "runner=dask_address",
            "runner.task_runner.address=tcp://dask-scheduler:8786",
        ],
    )
    plan = resolve_execution(cfg.runner)

    assert plan.family == "dask"
    assert plan.task_runner.kwargs["address"] == "tcp://dask-scheduler:8786"
    assert "cluster_kwargs" not in plan.task_runner.kwargs


def test_dask_cluster_runner_choice_carries_cluster_class_and_kwargs(monkeypatch):
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    fake_module = types.ModuleType("prefect_dask")

    class DaskTaskRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module.DaskTaskRunner = DaskTaskRunner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefect_dask", fake_module)
    monkeypatch.setattr(optional, "_PREFECT_DASK_AVAILABLE", True)

    cfg = _compose(
        "extract",
        overrides=[
            "runner=dask_cluster",
            "runner.task_runner.cluster_class=dask_kubernetes.operator.KubeCluster",
            "+runner.task_runner.cluster_kwargs.namespace=radiologist",
        ],
    )
    plan = resolve_execution(cfg.runner)

    assert plan.family == "dask"
    assert (
        plan.task_runner.kwargs["cluster_class"]
        == "dask_kubernetes.operator.KubeCluster"
    )
    assert plan.task_runner.kwargs["cluster_kwargs"]["namespace"] == "radiologist"


def test_build_config_yields_same_plan_shape_as_extract_config():
    from radiologist.etl.execution import resolve_execution

    extract_cfg = _compose("extract")
    build_cfg = _compose("build")

    extract_plan = resolve_execution(extract_cfg.runner)
    build_plan = resolve_execution(build_cfg.runner)

    assert extract_plan.family == build_plan.family
    assert extract_plan.batch_size == build_plan.batch_size
    assert type(extract_plan.task_runner) is type(build_plan.task_runner)


def test_assign_split_config_rejects_runner_override():
    with pytest.raises(ConfigCompositionException):
        _compose("assign_split", overrides=["runner=local"])


def test_resolve_execution_with_no_config_yields_local_family():
    from radiologist.etl.execution import resolve_execution

    plan = resolve_execution(None)

    assert plan.family == "local"


def test_resolve_execution_with_unknown_family_raises_value_error():
    from radiologist.etl.execution import resolve_execution

    with pytest.raises(
        ValueError, match="local.*dask.*ray.*beam|beam.*dask.*ray.*local"
    ):
        resolve_execution({"family": "spark"})


def test_resolve_execution_dask_without_backend_installed_raises_runtime_error(
    monkeypatch,
):
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    # #186 needed a real prefect_dask install in this shared environment to
    # exercise real DaskTaskRunner wiring, so the backend is genuinely
    # available here now — force the unavailable branch explicitly instead
    # of relying on real absence.
    monkeypatch.setattr(optional, "_PREFECT_DASK_AVAILABLE", False)

    cfg = _compose("extract", overrides=["runner=dask_local"])

    with pytest.raises(RuntimeError, match="dask"):
        resolve_execution(cfg.runner)


def test_resolve_execution_beam_family_builds_beam_executor(monkeypatch):
    from radiologist.etl import optional
    from radiologist.etl.beam_executor import BeamExecutor
    from radiologist.etl.execution import resolve_execution

    monkeypatch.setattr(optional, "_BEAM_AVAILABLE", True)

    cfg = _compose(
        "extract",
        overrides=[
            "runner=beam_direct",
            "runner.beam.parts_dir=/tmp/beam-parts",
        ],
    )
    plan = resolve_execution(cfg.runner)

    assert plan.family == "beam"
    assert plan.task_runner is None
    assert isinstance(plan.beam, BeamExecutor)


def test_resolve_execution_beam_without_backend_installed_raises_runtime_error(
    monkeypatch,
):
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    # #189 needed a real apache-beam install in this shared environment to
    # exercise the real BeamExecutor against the direct runner, so the
    # backend is genuinely available here now — force the unavailable branch
    # explicitly instead of relying on real absence, exactly as the dask test
    # above does.
    monkeypatch.setattr(optional, "_BEAM_AVAILABLE", False)

    cfg = _compose(
        "extract",
        overrides=[
            "runner=beam_direct",
            "runner.beam.parts_dir=/tmp/beam-parts",
        ],
    )

    with pytest.raises(RuntimeError, match="beam"):
        resolve_execution(cfg.runner)


def test_batch_size_argument_overrides_runner_config_default():
    from radiologist.etl.execution import resolve_execution

    cfg = _compose("extract")
    plan = resolve_execution(cfg.runner, batch_size=128)

    assert plan.batch_size == 128


def test_batch_size_defaults_to_runner_config_value():
    from radiologist.etl.execution import resolve_execution

    cfg = _compose("extract")
    plan = resolve_execution(cfg.runner)

    assert plan.batch_size == 64


@pytest.mark.parametrize("runner_choice", RUNNER_CHOICES)
def test_every_shipped_runner_choice_composes_under_hydra(runner_choice):
    cfg = _compose("extract", overrides=[f"runner={runner_choice}"])

    assert cfg.runner.family in {"local", "dask", "ray", "beam"}


# --- default_workers --------------------------------------------------------


def test_default_workers_returns_a_positive_int():
    import os

    from radiologist.etl.execution import default_workers

    workers = default_workers()

    assert isinstance(workers, int)
    assert workers >= 1
    assert workers == (os.cpu_count() or 1)


# --- chunked -----------------------------------------------------------------


def test_chunked_splits_sequence_into_consecutive_chunks_of_at_most_size():
    from radiologist.etl.execution import chunked

    result = chunked([1, 2, 3, 4, 5], 2)

    assert result == [[1, 2], [3, 4], [5]]


def test_chunked_returns_empty_list_for_empty_input():
    from radiologist.etl.execution import chunked

    assert chunked([], 3) == []


def test_chunked_raises_value_error_for_size_below_one():
    from radiologist.etl.execution import chunked

    with pytest.raises(ValueError):
        chunked([1, 2, 3], 0)


# --- local_mapper --------------------------------------------------------------


def test_local_mapper_applies_fn_to_each_item_and_preserves_order():
    from radiologist_etl_tests._picklable_fns import _double

    from radiologist.etl.execution import local_mapper

    mapper = local_mapper(_double, workers=2)

    assert mapper([1, 2, 3, 4]) == [2, 4, 6, 8]


def test_local_mapper_returns_empty_list_for_empty_input():
    from radiologist_etl_tests._picklable_fns import _double

    from radiologist.etl.execution import local_mapper

    mapper = local_mapper(_double, workers=2)

    assert mapper([]) == []


# --- storage_options_from_cfg ---------------------------------------------------


def test_storage_options_from_cfg_returns_none_when_key_absent():
    from omegaconf import OmegaConf

    from radiologist.etl.execution import storage_options_from_cfg

    cfg = OmegaConf.create({"foo": "bar"})

    assert storage_options_from_cfg(cfg) is None


def test_storage_options_from_cfg_returns_none_when_explicitly_null():
    from omegaconf import OmegaConf

    from radiologist.etl.execution import storage_options_from_cfg

    cfg = OmegaConf.create({"storage_options": None})

    assert storage_options_from_cfg(cfg) is None


def test_storage_options_from_cfg_treats_explicit_empty_dict_as_a_valid_value():
    from omegaconf import OmegaConf

    from radiologist.etl.execution import storage_options_from_cfg

    cfg = OmegaConf.create({"storage_options": {}})

    assert storage_options_from_cfg(cfg) == {}


def test_storage_options_from_cfg_returns_plain_dict_for_populated_options():
    from omegaconf import OmegaConf

    from radiologist.etl.execution import storage_options_from_cfg

    cfg = OmegaConf.create({"storage_options": {"key": "abc123"}})

    assert storage_options_from_cfg(cfg) == {"key": "abc123"}


# --- the shipped default runner works on a plain, no-extras install -------------


def test_shipped_default_runner_resolves_without_prefect_installed(monkeypatch):
    """The ``local`` family is unconditionally available: the shipped default
    configuration must resolve on an install carrying no optional extras."""
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    monkeypatch.setattr(optional, "_PREFECT_AVAILABLE", False)

    cfg = _compose("extract")
    plan = resolve_execution(cfg.runner)

    assert plan.family == "local"


def test_shipped_default_runner_carries_no_task_runner_without_prefect(monkeypatch):
    """``conf/runner/local.yaml`` targets ``prefect.task_runners.
    ProcessPoolTaskRunner``. Without prefect that target cannot be
    instantiated, so a plan carrying no task runner is the observable proof
    that it was never evaluated."""
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    monkeypatch.setattr(optional, "_PREFECT_AVAILABLE", False)

    cfg = _compose("extract")
    plan = resolve_execution(cfg.runner)

    assert plan.task_runner is None


def test_shipped_build_default_runner_also_resolves_without_prefect(monkeypatch):
    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    monkeypatch.setattr(optional, "_PREFECT_AVAILABLE", False)

    plan = resolve_execution(_compose("build").runner)

    assert (plan.family, plan.task_runner) == ("local", None)


@pytest.mark.parametrize(
    ("runner_choice", "sentinel", "expected_extra"),
    [
        ("dask_local", "_PREFECT_DASK_AVAILABLE", "dask"),
        ("ray_local", "_PREFECT_RAY_AVAILABLE", "ray"),
        ("beam_direct", "_BEAM_AVAILABLE", "beam"),
    ],
)
def test_unavailable_backend_names_a_declared_extra_in_prose_and_command(
    monkeypatch, runner_choice, sentinel, expected_extra
):
    """Every family that can still be unavailable must offer a remedy the
    user can actually follow: the extra named in the prose, the extra named
    in the install command and an extra the package really declares must all
    be the same string."""
    from importlib.metadata import metadata

    from radiologist.etl import optional
    from radiologist.etl.execution import resolve_execution

    monkeypatch.setattr(optional, sentinel, False)

    overrides = [f"runner={runner_choice}"]
    if runner_choice == "beam_direct":
        overrides.append("runner.beam.parts_dir=/tmp/beam-parts")
    cfg = _compose("extract", overrides=overrides)

    with pytest.raises(RuntimeError) as excinfo:
        resolve_execution(cfg.runner)

    message = str(excinfo.value)
    declared = set(metadata("radiologist-etl").get_all("Provides-Extra") or [])
    assert expected_extra in declared
    assert f"the {expected_extra} extra is required" in message
    assert f"pip install 'radiologist-etl[{expected_extra}]'" in message


def test_explicitly_null_batch_size_falls_back_to_the_documented_default():
    from radiologist.etl.execution import resolve_execution

    cfg = _compose("extract", overrides=["runner.batch_size=null"])
    plan = resolve_execution(cfg.runner)

    assert plan.batch_size == 64


def _constant_stat(image, metadata, mask=None):
    """A trivial StatExtractor: this test is about batch sizing, not features."""
    return {"stat": 1.0}


def test_extract_stage_runs_under_a_plan_built_from_a_null_batch_size(
    tmp_path, image_dir
):
    """A null ``batch_size`` used to reach ``chunked()`` as ``None`` and blow
    up with a TypeError once a stage actually ran under the plan.

    The mapper is injected so the real ``process_batch`` runs in this process:
    ``chunked(paths, batch_size)`` is evaluated before any dispatch, so a
    process pool would add nothing to the assertion.
    """
    import functools

    from radiologist.etl.execution import resolve_execution
    from radiologist.etl.extract import extract
    from radiologist.etl.processors import process_batch

    plan = resolve_execution(
        _compose("extract", overrides=["runner.batch_size=null"]).runner
    )

    images = [str(p) for p in sorted(image_dir.rglob("*.png"))]
    listing = tmp_path / "listing.txt"
    listing.write_text("\n".join(images) + "\n")

    worker = functools.partial(
        process_batch,
        images_root=None,
        masks_root=None,
        manifest_id="placeholder",
        extractors=[_constant_stat],
    )

    result = extract(
        file_list=str(listing),
        destination=str(tmp_path / "manifests"),
        extractors=[_constant_stat],
        iqr_columns=["stat"],
        batch_size=plan.batch_size,
        mapper=lambda batches: [worker(batch) for batch in batches],
    )

    assert result.total == len(images)


def test_explicitly_null_family_still_yields_a_local_plan():
    """Regression pin on existing behaviour, unchanged by this fix."""
    from radiologist.etl.execution import resolve_execution

    cfg = _compose("extract", overrides=["runner.family=null"])
    plan = resolve_execution(cfg.runner)

    assert plan.family == "local"
