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

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf


def _make_artifact(qualified_name: str, version: str, run_id: str) -> MagicMock:
    art = MagicMock()
    art.qualified_name = qualified_name
    art.version = version
    source_run = MagicMock()
    source_run.id = run_id
    art.logged_by.return_value = source_run
    return art


class TestResolveResumeCkptFromScratch:
    def test_returns_none_when_all_keys_null(self) -> None:
        from radiologist.core.resume import resolve_resume_ckpt

        cfg = OmegaConf.create(
            {"ckpt_path": None, "resume_ref": None, "resume_path": None}
        )

        assert resolve_resume_ckpt(cfg) is None


class TestResolveResumeCkptFromLocalPath:
    def test_returns_ckpt_path_unchanged_when_no_resume_ref(
        self, ckpt_path: str
    ) -> None:
        from radiologist.core.resume import resolve_resume_ckpt

        cfg = OmegaConf.create(
            {"ckpt_path": ckpt_path, "resume_ref": None, "resume_path": None}
        )

        assert resolve_resume_ckpt(cfg) == ckpt_path


class TestResolveResumeCkptFromWandbRef:
    def test_resolves_and_downloads_via_wandb_registry(
        self, tmp_path: Any, ckpt_path: str
    ) -> None:
        from radiologist.core.resume import resolve_resume_ckpt

        art = _make_artifact("entity/project/model-run123:best", "v3", "run123")

        cfg = OmegaConf.create(
            {
                "ckpt_path": None,
                "resume_ref": "run123:best",
                "resume_path": "entity/project",
                "paths": {"output_dir": str(tmp_path)},
            }
        )

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            mock_api = MagicMock()
            mock_wandb.Api.return_value = mock_api
            mock_api.artifact.return_value = art
            art.download.return_value = str(tmp_path)

            result = resolve_resume_ckpt(cfg)

        assert result == ckpt_path
        mock_api.artifact.assert_any_call(
            type="model", name="entity/project/model-run123:best"
        )

    def test_raises_value_error_when_resume_path_missing(self) -> None:
        from radiologist.core.resume import resolve_resume_ckpt

        cfg = OmegaConf.create(
            {"ckpt_path": None, "resume_ref": "run123:best", "resume_path": None}
        )

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            with pytest.raises(ValueError, match="resume_ref.*resume_path"):
                resolve_resume_ckpt(cfg)
            mock_wandb.Api.assert_not_called()

    def test_raises_value_error_when_both_ckpt_path_and_resume_ref_set(
        self, ckpt_path: str
    ) -> None:
        from radiologist.core.resume import resolve_resume_ckpt

        cfg = OmegaConf.create(
            {
                "ckpt_path": ckpt_path,
                "resume_ref": "run123:best",
                "resume_path": "entity/project",
            }
        )

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            with pytest.raises(ValueError, match="ckpt_path.*resume_ref"):
                resolve_resume_ckpt(cfg)
            mock_wandb.Api.assert_not_called()

    @pytest.mark.parametrize(
        "resume_ref",
        [
            "run123best",
            "run123:",
            ":best",
            "run123:best:extra",
        ],
    )
    def test_raises_value_error_for_malformed_resume_ref(self, resume_ref: str) -> None:
        from radiologist.core.resume import resolve_resume_ckpt

        cfg = OmegaConf.create(
            {
                "ckpt_path": None,
                "resume_ref": resume_ref,
                "resume_path": "entity/project",
            }
        )

        with patch("radiologist.registry.resolver._wandb") as mock_wandb:
            with pytest.raises(ValueError):
                resolve_resume_ckpt(cfg)
            mock_wandb.Api.assert_not_called()


class TestRestorePrecision:
    def test_restores_precision_from_checkpoint(self, ckpt_path: str) -> None:
        from radiologist.core.resume import restore_precision

        cfg = OmegaConf.create({"trainer": {"precision": 32}})

        restore_precision(cfg, ckpt_path)

        stored = torch.load(ckpt_path, weights_only=False)
        assert cfg.trainer.precision == stored["precision"]

    def test_is_noop_when_checkpoint_has_no_precision_entry(
        self, tmp_path: Any, lmodule: Any
    ) -> None:
        from radiologist.core.resume import restore_precision

        path = str(tmp_path / "no_precision.ckpt")
        torch.save({"state_dict": lmodule.state_dict()}, path)

        cfg = OmegaConf.create({"trainer": {"precision": 32}})

        restore_precision(cfg, path)

        assert cfg.trainer.precision == 32
