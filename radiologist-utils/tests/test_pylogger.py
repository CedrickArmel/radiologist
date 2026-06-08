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
from unittest.mock import patch

import pytest

from radiologist.utils import RankedLogger


class TestRankedLoggerAutoDetectsRank:
    def test_log_without_rank_kwarg_emits_rank_zero_prefix_in_single_process(
        self, caplog
    ):
        logger = RankedLogger(rank_zero_only=False)
        with caplog.at_level(logging.DEBUG):
            logger.log(logging.INFO, "hello")
        assert any("[rank: 0]" in record.message for record in caplog.records)

    def test_rank_zero_only_does_not_raise_when_no_rank_kwarg_passed(self):
        logger = RankedLogger(rank_zero_only=True)
        try:
            logger.log(logging.INFO, "hello")
        except RuntimeError:
            pytest.fail(
                "RankedLogger raised RuntimeError when rank_zero_only=True "
                "and no rank= kwarg was passed"
            )

    def test_rank_zero_only_suppresses_message_when_detected_rank_is_nonzero(
        self, caplog
    ):
        logger = RankedLogger(rank_zero_only=True)
        with caplog.at_level(logging.DEBUG):
            with patch(
                "radiologist.utils.ml.pylogger._rank_zero_only",
                **{"rank": 1},
            ) as mock_rzo:
                mock_rzo.rank = 1
                logger.log(logging.INFO, "hello")
        assert len(caplog.records) == 0

    def test_rank_zero_only_emits_message_when_detected_rank_is_zero(self, caplog):
        logger = RankedLogger(rank_zero_only=True)
        with caplog.at_level(logging.DEBUG):
            logger.log(logging.INFO, "hello")
        assert len(caplog.records) > 0


class TestRankedLoggerPublicExports:
    def test_ranked_logger_importable_from_radiologist_utils(self):
        from radiologist.utils import RankedLogger as RL

        assert RL is not None

    def test_ranked_logger_importable_from_radiologist_utils_ml(self):
        from radiologist.utils.ml import RankedLogger as RL

        assert RL is not None

    def test_both_imports_resolve_to_same_class(self):
        from radiologist.utils import RankedLogger as RL1
        from radiologist.utils.ml import RankedLogger as RL2

        assert RL1 is RL2


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
