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
``radiologist-inference/src/radiologist/inference/cli.py``.
"""

from typing import List, Optional

import typer

app = typer.Typer(name="infer", add_completion=False)

__all__ = ["app", "run"]


@app.command()
def predict(
    image_path: str = typer.Argument(..., help="Path to the input chest X-ray image."),
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to the deterministic ONNX model."
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Mutually exclusive with --path.",
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
        "into. Ignored when --path is used.",
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
    raise NotImplementedError


@app.command()
def explain(
    image_path: str = typer.Argument(..., help="Path to the input chest X-ray image."),
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to the deterministic ONNX model."
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Mutually exclusive with --path.",
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
        "into. Ignored when --path is used.",
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
    raise NotImplementedError


@app.command()
def uncertainty(
    image_path: str = typer.Argument(..., help="Path to the input chest X-ray image."),
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to the MC-Dropout ONNX model."
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Mutually exclusive with --path.",
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
        "into. Ignored when --path is used.",
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
    raise NotImplementedError


@app.command()
def serve(
    path: Optional[str] = typer.Option(
        None, "--path", help="Path to the deterministic ONNX model."
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="W&B run ID identifying the registry artifact to resolve. "
        "Mutually exclusive with --path.",
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
        "into. Ignored when --path is used.",
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
    raise NotImplementedError


def run(argv: List[str]) -> int:
    """Run the ``infer`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``infer``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    raise NotImplementedError
