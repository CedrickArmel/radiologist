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

from typing import Optional

from omegaconf import DictConfig


def resolve_resume_ckpt(cfg: DictConfig) -> Optional[str]:
    """Resolve the checkpoint path to resume training from, if any.

    Precedence: ``resume_ref`` (W&B, needs ``resume_path``, exclusive with
    ``ckpt_path``) > ``ckpt_path`` (local path, unchanged) > ``None``
    (train from scratch).

    Raises:
        ValueError: if ``resume_ref`` is set without ``resume_path``, if both
            ``resume_ref`` and ``ckpt_path`` are set, or if ``resume_ref``
            does not parse as ``'<run_id>:<tag>'``.
    """
    raise NotImplementedError


def restore_precision(cfg: DictConfig, ckpt_path: str) -> None:
    """Restore ``cfg.trainer.precision`` from a checkpoint's stored value, if present.

    Security: loads with weights_only=False — Lightning checkpoints unpickle
    non-tensor objects (consistent with module.py / train.py). The trust
    boundary matches fit's own ckpt load: checkpoints originate from the
    user's own local path or their authenticated W&B project, never an
    untrusted source. Do not switch to weights_only=True.
    """
    raise NotImplementedError
