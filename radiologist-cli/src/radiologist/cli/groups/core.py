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

"""``radiologist core`` command group — Hydra-composed training entry point."""

import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig

from radiologist.core.train import train
from radiologist.utils.cli import emit, exit_code_for
from radiologist.utils.ml import get_metric_value

try:
    import wandb as _wandb
except ImportError:  # pragma: no cover - exercised via sentinel patching
    _wandb = None  # type: ignore[assignment]

__all__ = ["train_main", "run"]


def _assemble_record(object_dict: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """Build the fixed six-key result record from a completed ``train()`` call.

    The det/mcd ONNX paths and registry qualified names are not returned by
    ``train()`` -- they are a side effect of ``OnnxExportCallback.on_fit_end``
    (opt-in, silent no-op without an active W&B run or a best checkpoint).
    Reconstructed here from the same naming convention the callback and the
    registry uploader use (``model-{run_id}.onnx`` / ``model-{run_id}-mcd.onnx``,
    ``{entity}/{project}/model-{run_id}[-mcd]:best``), gated on the exported
    file actually existing on disk -- this is the accepted CLI/core coupling
    documented in issue #175's design notes.
    """
    trainer = object_dict.get("trainer")

    best_ckpt_path: Optional[str] = None
    if trainer is not None:
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_ckpt_path = getattr(checkpoint_callback, "best_model_path", "") or None

    run_obj = getattr(_wandb, "run", None) if _wandb is not None else None
    run_id: Optional[str] = (
        getattr(run_obj, "id", None) if run_obj is not None else None
    )

    det_onnx_path: Optional[str] = None
    mcd_onnx_path: Optional[str] = None
    det_qualified_name: Optional[str] = None
    mcd_qualified_name: Optional[str] = None

    if trainer is not None and run_id:
        out_dir = trainer.log_dir or trainer.default_root_dir
        entity = getattr(run_obj, "entity", "")
        project = getattr(run_obj, "project", "")

        candidate_det = os.path.join(out_dir, f"model-{run_id}.onnx")
        if os.path.exists(candidate_det):
            det_onnx_path = candidate_det
            det_qualified_name = f"{entity}/{project}/model-{run_id}:best"

        candidate_mcd = os.path.join(out_dir, f"model-{run_id}-mcd.onnx")
        if os.path.exists(candidate_mcd):
            mcd_onnx_path = candidate_mcd
            mcd_qualified_name = f"{entity}/{project}/model-{run_id}-mcd:best"

    return OrderedDict(
        (
            ("run_id", run_id),
            ("best_ckpt_path", best_ckpt_path),
            ("det_onnx_path", det_onnx_path),
            ("mcd_onnx_path", mcd_onnx_path),
            ("det_qualified_name", det_qualified_name),
            ("mcd_qualified_name", mcd_qualified_name),
        )
    )


@hydra.main(
    config_path="pkg://radiologist.core.configs",
    config_name="train",
    version_base="1.3",
)
def train_main(cfg: DictConfig) -> Optional[float]:
    """Hydra-composed entry point for the ``radiologist core`` command group.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.

    Returns:
        The scalar value of ``cfg.optimized_metric``, or ``None`` when unset.
    """
    metric_dict, object_dict = train(cfg)
    record = _assemble_record(object_dict)
    emit(record)

    optimized_metric = cfg.get("optimized_metric")
    if not optimized_metric:
        return None
    return get_metric_value(metric_dict, optimized_metric)


def run(argv: List[str]) -> int:
    """Run the ``core`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``core``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    old_argv = sys.argv
    old_full_error = os.environ.get("HYDRA_FULL_ERROR")
    sys.argv = ["radiologist-core-train", *argv]
    # Without this, Hydra's own run_and_report() swallows every exception
    # raised inside train_main() and always exits 1, defeating the
    # exit-code-per-exception-type contract below.
    os.environ["HYDRA_FULL_ERROR"] = "1"
    try:
        train_main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except Exception as exc:  # noqa: BLE001 - mapped to a process exit code
        print(f"Error: {exc}", file=sys.stderr)
        return exit_code_for(exc)
    finally:
        sys.argv = old_argv
        if old_full_error is None:
            os.environ.pop("HYDRA_FULL_ERROR", None)
        else:
            os.environ["HYDRA_FULL_ERROR"] = old_full_error
    return 0
