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

Entry points: predict <image> --model <det_path>
              explain <image> --model <det_path>
              uncertainty <image> --model <det_path> --mcd-model <mcd_path>
"""

from typing import Optional

import numpy as np

from radiologist.inference.classifier import Classifier
from radiologist.inference.explainer import Explainer
from radiologist.inference.mc_dropout import MCDropoutPredictor
from radiologist.inference.optional import _typer

if _typer is not None:
    import typer

    app = typer.Typer(name="radiologist", add_completion=False)

    @app.command()
    def predict(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
        model: str = typer.Option(
            ..., "--model", help="Path to the deterministic ONNX model."
        ),
    ) -> None:
        """Run classification inference on a chest X-ray image."""
        try:
            classifier = Classifier.from_path(det_path=model)
            result = classifier.predict(image=image_path)
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Predicted class: {result.predicted_class}")
        for cls, prob in result.probabilities.items():
            typer.echo(f"  {cls}: {prob:.4f}")

    @app.command()
    def explain(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
        model: str = typer.Option(
            ..., "--model", help="Path to the deterministic ONNX model."
        ),
        out: Optional[str] = typer.Option(
            None, "--out", help="Path to save the saliency map as a .npy file."
        ),
    ) -> None:
        """Produce a Score-CAM explanation for a chest X-ray image."""
        try:
            explainer = Explainer.from_path(det_path=model)
            result = explainer.explain(image=image_path)
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Predicted class: {result.predicted_class}")
        if out is not None:
            np.save(out, result.saliency_map)
            typer.echo(f"Saliency map saved to: {out}")
        else:
            typer.echo(f"Saliency map shape: {result.saliency_map.shape}")

    @app.command()
    def uncertainty(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
        model: str = typer.Option(
            ..., "--model", help="Path to the deterministic ONNX model."
        ),
        mcd_model: str = typer.Option(
            ..., "--mcd-model", help="Path to the MC-Dropout ONNX model."
        ),
        n_passes: int = typer.Option(
            30, "--n-passes", help="Number of stochastic forward passes."
        ),
    ) -> None:
        """Estimate MC-Dropout uncertainty for a chest X-ray image."""
        try:
            predictor = MCDropoutPredictor.from_path(det_path=model, mcd_path=mcd_model)
            result = predictor.predict_with_uncertainty(
                image=image_path, n_passes=n_passes
            )
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        typer.echo("Mean probabilities:")
        for cls, prob in result.mean_probabilities.items():
            std = result.std_per_class[cls]
            typer.echo(f"  {cls}: {prob:.4f} (std={std:.4f})")
        typer.echo(f"Predictive entropy: {result.predictive_entropy:.4f}")

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
