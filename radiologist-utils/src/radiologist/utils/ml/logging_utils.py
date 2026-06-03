# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
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

from lightning_utilities.core.rank_zero import (
    rank_zero_only,  # type: ignore[import-untyped]
)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Log hyperparameters from cfg/module/trainer to every trainer logger.

    Args:
        object_dict: Must contain keys "cfg" (DictConfig), "module"
            (LightningModule), and "trainer" (Trainer). No-op when the
            trainer has no logger attached.
    """
    trainer = object_dict.get("trainer")
    if not trainer or not trainer.logger:
        return

    hparams: dict = {}

    cfg = object_dict.get("cfg")
    if cfg is not None:
        hparams["cfg"] = cfg

    module = object_dict.get("module")
    if module is not None:
        hparams["module"] = type(module).__name__

    trainer.logger.log_hyperparams(hparams)
