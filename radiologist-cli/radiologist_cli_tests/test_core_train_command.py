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

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

RECORD_KEYS = (
    "run_id",
    "best_ckpt_path",
    "det_onnx_path",
    "mcd_onnx_path",
    "det_qualified_name",
    "mcd_qualified_name",
)


def _parse_kv_record(out: str) -> dict:
    """Parse only the ``key=value`` lines belonging to the fixed record.

    Ignores any incidental stdout noise emitted by the real training stack
    (e.g. third-party library prints) that doesn't match a record key.
    """
    lines = {}
    for line in out.strip().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in RECORD_KEYS:
            lines[key] = value
    return lines


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_PKGS = (
    "radiologist-utils",
    "radiologist-etl",
    "radiologist-core",
    "radiologist-inference",
    "radiologist-registry",
    "radiologist-cli",
)


def _subprocess_env(**extra: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / pkg / "src") for pkg in _WORKSPACE_PKGS
    )
    env.update(extra)
    return env


def _write_tiny_config_dir(
    tmp_path: Path, shard_root: Path, split_manifest_uri: str
) -> Path:
    """Real Hydra config-group yaml files, added via ``--config-dir``.

    Mirrors ``base_cfg`` (tiny real net + tiny real WebDataset shards) so the
    CLI-argv path can compose a fully real, cheap-to-run config without
    touching production ``module``/``datamodule`` defaults (which point at
    real data on disk).
    """
    cfg_dir = tmp_path / "hydra_confs"
    for group in ("module", "datamodule", "trainer", "callbacks"):
        (cfg_dir / group).mkdir(parents=True)

    module_cfg = {
        "net": {
            "_target_": "torch.nn.Sequential",
            "_args_": [
                {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "padding": 1,
                },
                {"_target_": "torch.nn.ReLU"},
                {"_target_": "torch.nn.Dropout", "p": 0.5},
                {"_target_": "torch.nn.AdaptiveAvgPool2d", "output_size": [1, 1]},
                {"_target_": "torch.nn.Flatten"},
                {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 2},
            ],
        },
        "loss": {"_target_": "radiologist.core.FocalLoss"},
        "metric": {
            "_target_": "torchmetrics.classification.MulticlassFBetaScore",
            "_partial_": True,
            "beta": 1.0,
            "num_classes": 2,
        },
        "optimizer": {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3},
        "scheduler": None,
        "trainable_layers": None,
        "priors": None,
    }
    OmegaConf.save(OmegaConf.create(module_cfg), str(cfg_dir / "module" / "tiny.yaml"))

    transform = {
        "_target_": "torchvision.transforms.Compose",
        "transforms": [
            {"_target_": "torchvision.transforms.Resize", "size": [8, 8]},
            {"_target_": "torchvision.transforms.ToTensor"},
        ],
    }
    loader_partial = {
        "_target_": "webdataset.WebLoader",
        "_partial_": True,
        "batch_size": None,
        "num_workers": 0,
    }
    datamodule_cfg = {
        "_target_": "radiologist.core.WebDatasetDataModule",
        "shard_root": str(shard_root),
        "split_manifest_uri": split_manifest_uri,
        "label_map": {"NORMAL": "normal", "ABNORMAL": "abnormal"},
        "train_transform": transform,
        "eval_transform": transform,
        "train_loader": loader_partial,
        "eval_loader": loader_partial,
        "batch_size": 2,
        "classes": ["abnormal", "normal"],
    }
    OmegaConf.save(
        OmegaConf.create(datamodule_cfg), str(cfg_dir / "datamodule" / "tiny.yaml")
    )

    OmegaConf.save(OmegaConf.create(None), str(cfg_dir / "callbacks" / "tiny.yaml"))

    return cfg_dir


# ``trainer`` is a bare (non-group) default entry in train.yaml -- there is
# no ``trainer/<option>.yaml`` swap mechanism for it, so its production
# values are overridden individually via dotted keys instead.
_TRAINER_OVERRIDES = (
    "trainer.max_epochs=1",
    "trainer.min_epochs=1",
    "+trainer.limit_train_batches=2",
    "+trainer.limit_val_batches=2",
    "trainer.accelerator=cpu",
    "trainer.devices=1",
    "trainer.precision=32-true",
    "trainer.deterministic=false",
    "trainer.gradient_clip_val=null",
    "+trainer.enable_progress_bar=false",
    "+trainer.enable_model_summary=false",
    "trainer.use_distributed_sampler=false",
)


def _run_cli(argv, cwd=None, timeout=120, env_extra=None):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from radiologist.cli.groups.core import run; "
            "sys.exit(run(sys.argv[1:]))",
            *argv,
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(**(env_extra or {})),
        cwd=cwd,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Fixed record schema
# ---------------------------------------------------------------------------


def test_completed_run_emits_full_record_with_onnx_and_registry_fields(
    base_cfg, capsys
):
    import wandb
    from radiologist.cli.groups.core import train_main

    fake_run = MagicMock()
    fake_run.id = "run123"
    fake_run.entity = "ent"
    fake_run.project = "proj"

    with patch.object(wandb, "run", fake_run):
        train_main(base_cfg)

    out = capsys.readouterr().out
    lines = _parse_kv_record(out)
    assert set(lines) == set(RECORD_KEYS)
    assert lines["run_id"] == "run123"
    assert lines["best_ckpt_path"]
    assert lines["det_onnx_path"]
    assert lines["mcd_onnx_path"]
    assert lines["det_qualified_name"] == "ent/proj/model-run123:best"
    assert lines["mcd_qualified_name"] == "ent/proj/model-run123-mcd:best"


def test_all_six_keys_present_when_no_onnx_export_and_no_registry(base_cfg, capsys):
    from radiologist.cli.groups.core import train_main

    cfg = OmegaConf.merge(base_cfg, {"callbacks": None})

    train_main(cfg)

    out = capsys.readouterr().out
    lines = _parse_kv_record(out)
    assert set(lines) == set(RECORD_KEYS)
    assert lines["run_id"] == ""
    assert lines["det_onnx_path"] == ""
    assert lines["mcd_onnx_path"] == ""
    assert lines["det_qualified_name"] == ""
    assert lines["mcd_qualified_name"] == ""


def test_run_without_test_stage_still_emits_full_key_set(base_cfg, capsys):
    from radiologist.cli.groups.core import train_main

    cfg = OmegaConf.merge(base_cfg, {"test": False, "callbacks": None})

    train_main(cfg)

    out = capsys.readouterr().out
    lines = _parse_kv_record(out)
    assert set(lines) == set(RECORD_KEYS)


# ---------------------------------------------------------------------------
# Optimized metric return contract
# ---------------------------------------------------------------------------


def test_returns_optimized_metric_value_when_configured(base_cfg):
    from radiologist.cli.groups.core import train_main

    cfg = OmegaConf.merge(base_cfg, {"callbacks": None})
    cfg.optimized_metric = "val_score"

    result = train_main(cfg)

    assert isinstance(result, float)


def test_returns_none_when_no_optimized_metric_configured(base_cfg):
    from radiologist.cli.groups.core import train_main

    cfg = OmegaConf.merge(base_cfg, {"callbacks": None})

    result = train_main(cfg)

    assert result is None


# ---------------------------------------------------------------------------
# Exit codes -- exercised through the real argv path via a tiny config-dir
# added on the Hydra search path (production module/datamodule defaults
# point at real, unavailable data and are never touched).
# ---------------------------------------------------------------------------


def test_missing_checkpoint_exits_2(tmp_path, shard_root, split_manifest_uri):
    cfg_dir = _write_tiny_config_dir(tmp_path, shard_root, split_manifest_uri)
    result = _run_cli(
        [
            f"--config-dir={cfg_dir}",
            "module=tiny",
            "datamodule=tiny",
            "callbacks=tiny",
            "module.metric.num_classes=2",
            *_TRAINER_OVERRIDES,
            "test=false",
            "ckpt_path=/does/not/exist.ckpt",
        ]
    )

    assert result.returncode == 2, result.stderr


def test_training_failure_exits_1_with_error_on_stderr(
    tmp_path, shard_root, split_manifest_uri
):
    cfg_dir = _write_tiny_config_dir(tmp_path, shard_root, split_manifest_uri)
    result = _run_cli(
        [
            f"--config-dir={cfg_dir}",
            "module=tiny",
            "datamodule=tiny",
            "callbacks=tiny",
            "module.metric.num_classes=2",
            *_TRAINER_OVERRIDES,
            "test=false",
            "module.net._target_=this.does.not.Exist",
        ]
    )

    assert result.returncode == 1, result.stdout
    assert any(line.startswith("Error: ") for line in result.stderr.splitlines())


# ---------------------------------------------------------------------------
# --output=json
# ---------------------------------------------------------------------------


def test_output_json_produces_one_parseable_object_with_six_keys():
    result = _run_cli(
        ["train=false", "test=false", "optimized_metric=null"],
        env_extra={"RADIOLOGIST_OUTPUT": "json"},
    )

    assert result.returncode == 0, result.stderr
    last_line = [line for line in result.stdout.splitlines() if line.strip()][-1]
    payload = json.loads(last_line)
    assert set(payload) == set(RECORD_KEYS)


# ---------------------------------------------------------------------------
# Hydra CLI surface: --help, key=value override, group override
# ---------------------------------------------------------------------------


def test_help_exits_0_and_prints_composed_config_tree():
    result = _run_cli(["--help"])

    assert result.returncode == 0
    assert "== Config ==" in result.stdout or "optimized_metric" in result.stdout


def test_key_value_override_is_reflected_in_composed_config():
    with initialize_config_module(
        config_module="radiologist.core.configs", version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=["seed=123"])
    assert cfg.seed == 123


def test_group_override_selecting_non_default_module_is_reflected(
    tmp_path, shard_root, split_manifest_uri
):
    cfg_dir = _write_tiny_config_dir(tmp_path, shard_root, split_manifest_uri)
    with initialize_config_module(
        config_module="radiologist.core.configs", version_base="1.3"
    ):
        default_cfg = compose(config_name="train")
        overridden_cfg = compose(
            config_name="train",
            overrides=[
                f"hydra.searchpath=[file://{cfg_dir}]",
                "module=tiny",
            ],
        )
    assert default_cfg.module != overridden_cfg.module
    assert overridden_cfg.module.net._target_ == "torch.nn.Sequential"
