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

import logging

import pytest

from radiologist.utils import RankedLogger


class TestRankedLoggerPrefixesRank:
    def test_message_is_emitted_with_rank_prefix(self, caplog):
        logger = RankedLogger(rank_zero_only=False)
        with caplog.at_level(logging.DEBUG):
            logger.log(logging.INFO, "hello", rank=2)
        assert any("2" in record.message for record in caplog.records)

    def test_message_emitted_when_rank_zero_only_and_rank_is_zero(self, caplog):
        logger = RankedLogger(rank_zero_only=True)
        with caplog.at_level(logging.DEBUG):
            logger.log(logging.INFO, "hello", rank=0)
        assert len(caplog.records) > 0

    def test_message_suppressed_when_rank_zero_only_and_rank_is_nonzero(self, caplog):
        logger = RankedLogger(rank_zero_only=True)
        with caplog.at_level(logging.DEBUG):
            logger.log(logging.INFO, "hello", rank=1)
        assert len(caplog.records) == 0

    def test_raises_runtime_error_when_rank_zero_only_and_rank_is_unset(self):
        logger = RankedLogger(rank_zero_only=True)
        with pytest.raises(RuntimeError):
            logger.log(logging.INFO, "hello")


class TestRankedLoggerPublicExport:
    def test_ranked_logger_importable_from_radiologist_utils(self):
        from radiologist.utils import RankedLogger as RL

        assert RL is not None


class TestLogHyperparameters:
    def test_no_op_when_trainer_has_no_logger(self):
        from unittest.mock import MagicMock

        from lightning_utilities.core.rank_zero import (  # type: ignore[import-untyped]
            rank_zero_only,
        )

        from radiologist.utils.ml import log_hyperparameters

        rank_zero_only.rank = 0

        trainer = MagicMock()
        trainer.logger = False

        object_dict = {"cfg": MagicMock(), "model": MagicMock(), "trainer": trainer}
        log_hyperparameters(object_dict)
