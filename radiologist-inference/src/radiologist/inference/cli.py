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

"""Typer-based CLI for radiologist-inference (issue #82).

Entry points: predict <image> --model <det_path>
              pull <artifact> --local-dir <dir>
"""

from typing import Optional

from radiologist.inference.optional import _typer
from radiologist.inference.predictor import Predictor, pull_model

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
        mcd_model: Optional[str] = typer.Option(
            None, "--mcd-model", help="Path to the MC-Dropout ONNX model."
        ),
    ) -> None:
        """Run classification inference on a chest X-ray image."""
        try:
            predictor = Predictor.from_path(det_path=model, mcd_path=mcd_model)
            result = predictor.predict(image=image_path)
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Predicted class: {result.predicted_class}")
        for cls, prob in result.probabilities.items():
            typer.echo(f"  {cls}: {prob:.4f}")

    @app.command()
    def pull(
        artifact: str = typer.Argument(
            ..., help="W&B artifact path (entity/project/name:version)."
        ),
        local_dir: str = typer.Option(
            ".", "--local-dir", help="Local directory to download the model to."
        ),
    ) -> None:
        """Download an ONNX model from the W&B Model Registry."""
        try:
            path = pull_model(artifact_path=artifact, local_dir=local_dir)
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Model downloaded to: {path}")

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
