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

"""``radiologist infer`` command group — Typer app fronting ONNX inference.

Commands: predict, explain, uncertainty, serve. Grammar carried over
verbatim from the deleted
``radiologist-inference/src/radiologist/inference/cli.py``. Bodies route
their keyed records through :func:`radiologist.utils.cli.emit`, and use the
repo-wide :func:`radiologist.cli.errors.exit_on_error` decorator and
:func:`radiologist.cli.errors.run_typer_app` runner.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import typer

from radiologist.cli.errors import exit_on_error, run_typer_app
from radiologist.inference import optional as _inference_optional
from radiologist.inference import verbs
from radiologist.inference.app import create_app
from radiologist.utils.cli import emit

if TYPE_CHECKING:
    from radiologist.inference.base_predictor import BasePredictor

app = typer.Typer(name="infer", add_completion=False)

__all__ = ["app", "run"]

_SERVE_EXTRA_MISSING_MSG = (
    "The 'serve' extra is required to use the serve command. "
    "Install it with: pip install radiologist-inference[serve]"
)
_TOO_MANY_VERBS_MSG = "Choose at most one of --predict/--explain/--uncertainty."


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated string of ints, e.g. "1,3,224,224"."""
    if value is None:
        return None
    return [int(part) for part in value.split(",")]


def _provenance_record(
    predictor: Optional["BasePredictor"],
) -> Dict[str, Optional[str]]:
    """Build the reportable provenance record for a loaded predictor.

    Args:
        predictor: Loaded predictor instance, or ``None`` when no predictor
            was loaded (e.g. ``serve`` started without a source).

    Returns:
        A dict with keys ``model_qualified_name`` and ``model_version``, both
        ``None`` when the predictor carries no provenance (i.e. it was loaded
        from a local file path rather than a registry selector) or is absent.
    """
    ref = predictor.provenance if predictor is not None else None
    if ref is None:
        return {"model_qualified_name": None, "model_version": None}
    return {"model_qualified_name": ref.qualified_name, "model_version": ref.version}


@app.command()
@exit_on_error
def predict(
    image_path: str = typer.Argument(..., help="Path to the input chest X-ray image."),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Path to the deterministic ONNX model, or (with a registry "
        "selector flag) the entity/project prefix to resolve against.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Requires --path (supplies the entity/project to resolve against).",
    ),
    tags: Optional[List[str]] = typer.Option(
        None,
        "--tags",
        help="Registry artifact tag(s) to filter by when resolving via "
        "--run-id is not used. Repeatable.",
    ),
    groups: Optional[List[str]] = typer.Option(
        None,
        "--groups",
        help="Registry artifact group(s) to filter by when resolving "
        "without --run-id. Repeatable.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Metric name used to pick the best-scoring artifact among "
        "candidates matching --tags/--groups.",
    ),
    local_dir: str = typer.Option(
        ".",
        "--local-dir",
        help="Local directory the resolved registry artifact is downloaded "
        "into. Ignored when no registry selector is given.",
    ),
    mean: Optional[float] = typer.Option(
        None, "--mean", help="Normalization mean (requires --std)."
    ),
    std: Optional[float] = typer.Option(
        None, "--std", help="Normalization std (requires --mean)."
    ),
    input_shape: Optional[str] = typer.Option(
        None,
        "--input-shape",
        help="Fallback [N,C,H,W] as comma-separated ints, e.g. 1,3,224,224.",
    ),
) -> None:
    """Run classification inference on a chest X-ray image."""
    classifier = verbs.load_predictor(
        verbs.get_verb("predict"),
        path,
        run_id,
        tags,
        groups,
        metric,
        local_dir,
        mean=mean,
        std=std,
        input_shape=_parse_int_list(input_shape),
    )
    result = classifier.predict(image=image_path)
    emit(
        {
            "predicted_class": result.predicted_class,
            "probabilities": result.probabilities,
            **_provenance_record(classifier),
        }
    )


@app.command()
@exit_on_error
def explain(
    image_path: str = typer.Argument(..., help="Path to the input chest X-ray image."),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Path to the deterministic ONNX model, or (with a registry "
        "selector flag) the entity/project prefix to resolve against.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Requires --path (supplies the entity/project to resolve against).",
    ),
    tags: Optional[List[str]] = typer.Option(
        None,
        "--tags",
        help="Registry artifact tag(s) to filter by when resolving via "
        "--run-id is not used. Repeatable.",
    ),
    groups: Optional[List[str]] = typer.Option(
        None,
        "--groups",
        help="Registry artifact group(s) to filter by when resolving "
        "without --run-id. Repeatable.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Metric name used to pick the best-scoring artifact among "
        "candidates matching --tags/--groups.",
    ),
    local_dir: str = typer.Option(
        ".",
        "--local-dir",
        help="Local directory the resolved registry artifact is downloaded "
        "into. Ignored when no registry selector is given.",
    ),
    out: Optional[str] = typer.Option(
        None, "--out", help="Path to save the saliency map as a .npy file."
    ),
    mean: Optional[float] = typer.Option(
        None, "--mean", help="Normalization mean (requires --std)."
    ),
    std: Optional[float] = typer.Option(
        None, "--std", help="Normalization std (requires --mean)."
    ),
    input_shape: Optional[str] = typer.Option(
        None,
        "--input-shape",
        help="Fallback [N,C,H,W] as comma-separated ints, e.g. 1,3,224,224.",
    ),
) -> None:
    """Produce a Score-CAM explanation for a chest X-ray image."""
    explainer = verbs.load_predictor(
        verbs.get_verb("explain"),
        path,
        run_id,
        tags,
        groups,
        metric,
        local_dir,
        mean=mean,
        std=std,
        input_shape=_parse_int_list(input_shape),
    )
    result = explainer.explain(image=image_path)
    saliency_path: Optional[str] = None
    if out is not None:
        np.save(out, result.saliency_map)
        saliency_path = out
    emit(
        {
            "predicted_class": result.predicted_class,
            "saliency_shape": list(result.saliency_map.shape),
            "saliency_path": saliency_path,
            **_provenance_record(explainer),
        }
    )


@app.command()
@exit_on_error
def uncertainty(
    image_path: str = typer.Argument(..., help="Path to the input chest X-ray image."),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Path to the MC-Dropout ONNX model, or (with a registry "
        "selector flag) the entity/project prefix to resolve against.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Requires --path (supplies the entity/project to resolve against).",
    ),
    tags: Optional[List[str]] = typer.Option(
        None,
        "--tags",
        help="Registry artifact tag(s) to filter by when resolving via "
        "--run-id is not used. Repeatable.",
    ),
    groups: Optional[List[str]] = typer.Option(
        None,
        "--groups",
        help="Registry artifact group(s) to filter by when resolving "
        "without --run-id. Repeatable.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Metric name used to pick the best-scoring artifact among "
        "candidates matching --tags/--groups.",
    ),
    local_dir: str = typer.Option(
        ".",
        "--local-dir",
        help="Local directory the resolved registry artifact is downloaded "
        "into. Ignored when no registry selector is given.",
    ),
    n_passes: int = typer.Option(
        30, "--n-passes", help="Number of stochastic forward passes."
    ),
    mean: Optional[float] = typer.Option(
        None, "--mean", help="Normalization mean (requires --std)."
    ),
    std: Optional[float] = typer.Option(
        None, "--std", help="Normalization std (requires --mean)."
    ),
    input_shape: Optional[str] = typer.Option(
        None,
        "--input-shape",
        help="Fallback [N,C,H,W] as comma-separated ints, e.g. 1,3,224,224.",
    ),
) -> None:
    """Estimate MC-Dropout uncertainty for a chest X-ray image."""
    predictor = verbs.load_predictor(
        verbs.get_verb("uncertainty"),
        path,
        run_id,
        tags,
        groups,
        metric,
        local_dir,
        mean=mean,
        std=std,
        input_shape=_parse_int_list(input_shape),
    )
    result = predictor.predict_with_uncertainty(image=image_path, n_passes=n_passes)
    emit(
        {
            "predicted_class": max(
                result.mean_probabilities, key=lambda k: result.mean_probabilities[k]
            ),
            "n_passes": result.n_passes,
            "predictive_entropy": result.predictive_entropy,
            "mean_probabilities": result.mean_probabilities,
            "std_probabilities": result.std_per_class,
            **_provenance_record(predictor),
        }
    )


@app.command()
@exit_on_error
def serve(
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help="Path to the deterministic ONNX model, or (with a registry "
        "selector flag) the entity/project prefix to resolve against.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Requires --path (supplies the entity/project to resolve against).",
    ),
    tags: Optional[List[str]] = typer.Option(
        None,
        "--tags",
        help="Registry artifact tag(s) to filter by when resolving via "
        "--run-id is not used. Repeatable.",
    ),
    groups: Optional[List[str]] = typer.Option(
        None,
        "--groups",
        help="Registry artifact group(s) to filter by when resolving "
        "without --run-id. Repeatable.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Metric name used to pick the best-scoring artifact among "
        "candidates matching --tags/--groups.",
    ),
    local_dir: str = typer.Option(
        ".",
        "--local-dir",
        help="Local directory the resolved registry artifact is downloaded "
        "into. Ignored when no registry selector is given.",
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host interface to bind the HTTP server to."
    ),
    port: int = typer.Option(
        8000, "--port", help="TCP port to bind the HTTP server to."
    ),
    predict: bool = typer.Option(
        False, "--predict", help="Serve a Classifier (predict verb)."
    ),
    explain: bool = typer.Option(
        False, "--explain", help="Serve an Explainer (explain verb, default)."
    ),
    uncertainty: bool = typer.Option(
        False,
        "--uncertainty",
        help="Serve an MCDropoutPredictor (uncertainty verb).",
    ),
) -> None:
    """Launch the FastAPI inference server via uvicorn."""
    if _inference_optional._uvicorn is None:
        raise RuntimeError(_SERVE_EXTRA_MISSING_MSG)
    if sum([predict, explain, uncertainty]) > 1:
        raise ValueError(_TOO_MANY_VERBS_MSG)
    verb_name = "predict" if predict else "uncertainty" if uncertainty else "explain"
    has_source = path is not None or any([run_id, tags, groups, metric])
    predictor = (
        verbs.load_predictor(
            verbs.get_verb(verb_name),
            path,
            run_id,
            tags,
            groups,
            metric,
            local_dir,
        )
        if has_source
        else None
    )
    fastapi_app = create_app(predictor)
    emit(
        {
            "host": host,
            "port": port,
            "verb": verb_name,
            "model_path": path,
            "model_run_id": run_id,
            **_provenance_record(predictor),
        }
    )
    _inference_optional._uvicorn.run(fastapi_app, host=host, port=port)


def run(argv: List[str]) -> int:
    """Run the ``infer`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``infer``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    return run_typer_app(app, argv, prog_name="radiologist infer")
