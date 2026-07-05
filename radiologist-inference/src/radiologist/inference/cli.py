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

"""Typer-based CLI for radiologist-inference.

Entry points: predict <image> --path <det_path>
              explain <image> --path <det_path>
              uncertainty <image> --path <mcd_path>
"""

import functools
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np

from radiologist.inference import verbs
from radiologist.inference.app import create_app
from radiologist.inference.optional import _typer, _uvicorn

F = TypeVar("F", bound=Callable[..., None])


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated string of ints, e.g. "1,3,224,224"."""
    if value is None:
        return None
    return [int(part) for part in value.split(",")]


if _typer is not None:
    import typer

    app = typer.Typer(name="radiologist", add_completion=False)

    def _exit_on_error(func: F) -> F:
        """Wrap a CLI command so unhandled exceptions become a clean exit(1)."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            try:
                func(*args, **kwargs)
            except Exception as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(code=1)

        return wrapper  # type: ignore[return-value]

    @app.command()
    @_exit_on_error
    def predict(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
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
        """Run classification inference on a chest X-ray image.

        Loads a :class:`~radiologist.inference.classifier.Classifier` and
        prints the predicted class label and per-class probabilities.

        The model to load is dispatched via ``verbs.load_predictor``: if any
        registry selector flag (``--run-id``/``--tags``/``--groups``/
        ``--metric``) is set, the artifact is resolved and pulled from the
        W&B Registry into ``--local-dir``; otherwise ``--path`` is loaded
        directly from a local ONNX path. Exactly one of the two loading
        strategies must be usable, or the command exits with an error.

        Args:
            image_path: Path to the input chest X-ray image.
            path: Path to a local deterministic ONNX model file.
            run_id: W&B run ID for registry-backed resolution.
            tags: Registry artifact tags used for selector-based resolution.
            groups: Registry artifact groups used for selector-based
                resolution.
            metric: Metric name used to break ties among selector matches.
            local_dir: Directory to download registry artifacts into.
            mean: Optional normalization mean, requires std.
            std: Optional normalization std, requires mean.
            input_shape: Optional fallback [N, C, H, W] input shape.
        """
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
        typer.echo(f"Predicted class: {result.predicted_class}")
        for cls, prob in result.probabilities.items():
            typer.echo(f"  {cls}: {prob:.4f}")

    @app.command()
    @_exit_on_error
    def explain(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
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
        """Produce a Score-CAM explanation for a chest X-ray image.

        Loads an :class:`~radiologist.inference.explainer.Explainer` and
        prints the predicted class label, then either saves the saliency map
        to ``--out`` as a ``.npy`` file or prints its shape.

        Uses the same registry-selector-vs-local-path dispatch as
        ``predict``, via ``verbs.load_predictor``: registry selector flags
        (``--run-id``/``--tags``/``--groups``/``--metric``) resolve and pull
        the artifact into ``--local-dir``; otherwise ``--path`` is loaded
        directly.

        Args:
            image_path: Path to the input chest X-ray image.
            path: Path to a local deterministic ONNX model file.
            run_id: W&B run ID for registry-backed resolution.
            tags: Registry artifact tags used for selector-based resolution.
            groups: Registry artifact groups used for selector-based
                resolution.
            metric: Metric name used to break ties among selector matches.
            local_dir: Directory to download registry artifacts into.
            out: Optional path to save the saliency map as a .npy file. When
                omitted, only the saliency map's shape is printed.
            mean: Optional normalization mean, requires std.
            std: Optional normalization std, requires mean.
            input_shape: Optional fallback [N, C, H, W] input shape.
        """
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
        typer.echo(f"Predicted class: {result.predicted_class}")
        if out is not None:
            np.save(out, result.saliency_map)
            typer.echo(f"Saliency map saved to: {out}")
        else:
            typer.echo(f"Saliency map shape: {result.saliency_map.shape}")

    @app.command()
    @_exit_on_error
    def uncertainty(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
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
        """Estimate MC-Dropout uncertainty for a chest X-ray image.

        Loads an :class:`~radiologist.inference.mc_dropout.MCDropoutPredictor`
        (single-session: one ONNX model runs both the deterministic and the
        stochastic MC-Dropout forward passes), runs ``n_passes`` stochastic
        forward passes, and prints per-class mean probability with standard
        deviation plus the overall predictive entropy.

        Uses the same registry-selector-vs-local-path dispatch as
        ``predict``/``explain``, via ``verbs.load_predictor``: registry
        selector flags (``--run-id``/``--tags``/``--groups``/``--metric``)
        resolve and pull the artifact into ``--local-dir``; otherwise
        ``--path`` is loaded directly.

        Args:
            image_path: Path to the input chest X-ray image.
            path: Path to a local MC-Dropout ONNX model file.
            run_id: W&B run ID for registry-backed resolution.
            tags: Registry artifact tags used for selector-based resolution.
            groups: Registry artifact groups used for selector-based
                resolution.
            metric: Metric name used to break ties among selector matches.
            local_dir: Directory to download registry artifacts into.
            n_passes: Number of stochastic forward passes to average over.
            mean: Optional normalization mean, requires std.
            std: Optional normalization std, requires mean.
            input_shape: Optional fallback [N, C, H, W] input shape.
        """
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
        typer.echo("Mean probabilities:")
        for cls, prob in result.mean_probabilities.items():
            std = result.std_per_class[cls]
            typer.echo(f"  {cls}: {prob:.4f} (std={std:.4f})")
        typer.echo(f"Predictive entropy: {result.predictive_entropy:.4f}")

    @app.command()
    @_exit_on_error
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
        """Launch the FastAPI inference server via uvicorn.

        Loads a predictor matching exactly one of ``--predict``/``--explain``/
        ``--uncertainty`` (default ``explain``) using the same
        registry-selector-vs-local-path dispatch as the other commands: if a
        registry selector (``--run-id``/``--tags``/``--groups``/``--metric``)
        is set, the artifact is resolved and pulled into ``--local-dir``; if
        ``--path`` is set, it is loaded directly; if neither is set, the
        server starts with no predictor loaded and ``/predict``, ``/explain``,
        and ``/uncertainty`` respond with a 503 "no model loaded" error until
        a predictor becomes available. ``/healthz`` and ``/readyz`` are
        always available.

        Args:
            path: Path to a local ONNX model file.
            run_id: W&B run ID for registry-backed resolution.
            tags: Registry artifact tags used for selector-based resolution.
            groups: Registry artifact groups used for selector-based
                resolution.
            metric: Metric name used to break ties among selector matches.
            local_dir: Directory to download registry artifacts into.
            host: Host interface uvicorn binds the HTTP server to.
            port: TCP port uvicorn binds the HTTP server to.
            predict: Serve a Classifier (predict verb).
            explain: Serve an Explainer (explain verb, default).
            uncertainty: Serve an MCDropoutPredictor (uncertainty verb).

        Raises:
            ValueError: When more than one of --predict/--explain/
                --uncertainty is set.
        """
        if _uvicorn is None:
            raise RuntimeError(
                "The 'serve' extra is required to use the serve command. "
                "Install it with: pip install radiologist-inference[serve]"
            )
        if sum([predict, explain, uncertainty]) > 1:
            raise ValueError("Choose at most one of --predict/--explain/--uncertainty.")
        verb_name = (
            "predict" if predict else "uncertainty" if uncertainty else "explain"
        )
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
        _uvicorn.run(fastapi_app, host=host, port=port)

else:
    app = None  # type: ignore[assignment]


def main() -> None:
    """CLI entry point — raises RuntimeError when the cli extra is not installed."""
    if _typer is None:
        raise RuntimeError(
            "The 'cli' extra is required to use the radiologist CLI. "
            "Install it with: pip install radiologist-inference[cli]"
        )
    app()  # type: ignore[misc]
