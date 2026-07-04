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

"""Hydra/Lightning training utilities: instantiators, seeding, logging, nn init."""

from radiologist.utils.ml.hydra_utils import extras, get_metric_value, task_wrapper
from radiologist.utils.ml.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    sequential_scheduler,
)
from radiologist.utils.ml.logging_utils import log_hyperparameters
from radiologist.utils.ml.nn import initialize_weights
from radiologist.utils.ml.pylogger import RankedLogger
from radiologist.utils.ml.rich_utils import enforce_tags, print_config_tree
from radiologist.utils.ml.seeding import get_seeded_generator, seed_worker, set_seed

__all__ = [
    "enforce_tags",
    "extras",
    "get_metric_value",
    "get_seeded_generator",
    "initialize_weights",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "RankedLogger",
    "seed_worker",
    "sequential_scheduler",
    "set_seed",
    "task_wrapper",
]
