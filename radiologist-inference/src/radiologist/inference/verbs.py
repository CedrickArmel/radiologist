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

"""Predictor-verb registry binding CLI verb names to predictor classes."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from radiologist.inference.base_predictor import BasePredictor
from radiologist.inference.classifier import Classifier
from radiologist.inference.explainer import Explainer
from radiologist.inference.mc_dropout import MCDropoutPredictor
from radiologist.registry import selector_from_flags

_SELECTOR_REQUIRED_MSG = (
    "Provide either --path or a registry selector "
    "(--run-id/--tags/--groups/--metric)."
)
_MUTUALLY_EXCLUSIVE_MSG = (
    "Provide either --path or a registry selector "
    "(--run-id/--tags/--groups/--metric), not both."
)


@dataclass(frozen=True)
class PredictorVerb:
    """Immutable descriptor binding a CLI verb name to a predictor class.

    Attributes:
        name: CLI verb name (e.g. "predict", "explain", "uncertainty").
        predictor_cls: Predictor class the verb constructs.
        mcd_convention: Whether registry resolution applies the
            ``{run_id}-mcd`` naming convention for this verb.
    """

    name: str
    predictor_cls: Type[BasePredictor]
    mcd_convention: bool


_VERBS: Dict[str, PredictorVerb] = {
    v.name: v
    for v in (
        PredictorVerb("predict", Classifier, False),
        PredictorVerb("explain", Explainer, False),
        PredictorVerb("uncertainty", MCDropoutPredictor, True),
    )
}


def get_verb(name: str) -> PredictorVerb:
    """Return the registered ``PredictorVerb`` for ``name``.

    Args:
        name: Verb name to look up (e.g. ``"predict"``, ``"explain"``,
            ``"uncertainty"``).

    Returns:
        The registered ``PredictorVerb`` for ``name``.

    Raises:
        KeyError: If ``name`` is not a registered verb.
    """
    return _VERBS[name]


def apply_mcd_convention(run_id: Optional[str]) -> Optional[str]:
    """Encode the repo-wide det/mcd registry naming convention.

    Args:
        run_id: Deterministic model's run id, or ``None``.

    Returns:
        ``f"{run_id}-mcd"`` when ``run_id`` is truthy, else ``None``.
    """
    return f"{run_id}-mcd" if run_id else None


def load_predictor(
    verb: PredictorVerb,
    path: Optional[str],
    run_id: Optional[str],
    tags: Optional[List[str]],
    groups: Optional[List[str]],
    metric: Optional[str],
    local_dir: str,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    input_shape: Optional[List[int]] = None,
) -> BasePredictor:
    """Single loading path for every verb.

    When the (verb-adjusted) flags are registry-backed, resolves via
    ``verb.predictor_cls.from_selector(...)``; else when ``path`` is a local
    path, ``verb.predictor_cls.from_path(model_path=path)``; else raises
    ``ValueError`` naming ``--path`` and the registry selector flags.
    ``path`` and any registry selector flag are mutually exclusive: passing
    both raises ``ValueError`` rather than silently favoring one. When
    ``verb.mcd_convention`` is ``True``, ``run_id`` is rewritten via
    ``apply_mcd_convention`` before building the selector.

    Args:
        verb: Registered verb descriptor naming the predictor class to
            construct and whether the ``{run_id}-mcd`` convention applies.
        path: Local ONNX file path. Mutually exclusive with the registry
            selector flags (``run_id``/``tags``/``groups``/``metric``).
        run_id: Registry selector run id. Rewritten via
            ``apply_mcd_convention`` when ``verb.mcd_convention`` is ``True``.
        tags: Registry selector tags.
        groups: Registry selector groups.
        metric: Registry selector metric name.
        local_dir: Local directory where a registry-resolved ONNX file will
            be saved.
        mean: Optional normalization mean, forwarded unchanged to the
            predictor's ``from_selector``/``from_path`` constructor.
        std: Optional normalization std, forwarded unchanged to the
            predictor's ``from_selector``/``from_path`` constructor.
        input_shape: Optional input_shape fallback, forwarded unchanged to
            the predictor's ``from_selector``/``from_path`` constructor.

    Returns:
        Loaded predictor instance of ``verb.predictor_cls``.

    Raises:
        ValueError: If neither ``path`` nor a registry selector flag
            (``run_id``/``tags``/``groups``/``metric``) is provided, or if
            both ``path`` and a registry selector flag are provided.
    """
    effective_run_id = apply_mcd_convention(run_id) if verb.mcd_convention else run_id
    selector = selector_from_flags(
        path=path or "",
        run_id=effective_run_id,
        tags=tags,
        groups=groups,
        metric=metric,
    )
    has_path = path is not None
    registry_backed = selector.is_registry_backed()
    if has_path and registry_backed:
        raise ValueError(_MUTUALLY_EXCLUSIVE_MSG)
    if registry_backed:
        return verb.predictor_cls.from_selector(
            selector,
            local_dir=local_dir,
            mean=mean,
            std=std,
            input_shape=input_shape,
        )
    if has_path:
        return verb.predictor_cls.from_path(
            model_path=path, mean=mean, std=std, input_shape=input_shape
        )
    raise ValueError(_SELECTOR_REQUIRED_MSG)
