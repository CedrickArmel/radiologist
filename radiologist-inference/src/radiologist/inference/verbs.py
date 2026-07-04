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

from dataclasses import dataclass
from typing import List, Optional, Type

from radiologist.inference.base_predictor import BasePredictor


@dataclass(frozen=True)
class PredictorVerb:
    """Immutable descriptor binding a CLI verb name to the predictor class it
    constructs and whether registry resolution applies the ``{run_id}-mcd``
    convention."""

    name: str
    predictor_cls: Type[BasePredictor]
    mcd_convention: bool


def get_verb(name: str) -> PredictorVerb:
    """Return the registered ``PredictorVerb`` for ``name``.

    Raises ``KeyError`` for an unknown name.
    """
    raise NotImplementedError


def apply_mcd_convention(run_id: Optional[str]) -> Optional[str]:
    """Return ``f"{run_id}-mcd"`` when ``run_id`` is truthy, else ``None``.

    Encodes the repo-wide det/mcd registry pairing used by the uncertainty
    verb.
    """
    raise NotImplementedError


def load_predictor(
    verb: PredictorVerb,
    model: Optional[str],
    run_id: Optional[str],
    tags: Optional[List[str]],
    groups: Optional[List[str]],
    metric: Optional[str],
    local_dir: str,
) -> BasePredictor:
    """Single loading path for every verb.

    When the (verb-adjusted) flags are registry-backed, resolves via
    ``verb.predictor_cls.from_selector(...)``; else when ``model`` is a local
    path, ``verb.predictor_cls.from_path(model_path=model)``; else raises
    ``ValueError`` naming ``--model`` and the registry selector flags. When
    ``verb.mcd_convention`` is ``True``, ``run_id`` is rewritten via
    ``apply_mcd_convention`` before building the selector.
    """
    raise NotImplementedError
