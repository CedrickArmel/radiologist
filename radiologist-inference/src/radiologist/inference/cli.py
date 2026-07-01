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
        raise NotImplementedError

    @app.command()
    def explain(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
        model: str = typer.Option(
            ..., "--model", help="Path to the deterministic ONNX model."
        ),
    ) -> None:
        """Produce a Score-CAM explanation for a chest X-ray image."""
        raise NotImplementedError

    @app.command()
    def uncertainty(
        image_path: str = typer.Argument(
            ..., help="Path to the input chest X-ray image."
        ),
        mcd_model: str = typer.Option(
            ..., "--mcd-model", help="Path to the MC-Dropout ONNX model."
        ),
        n_passes: int = typer.Option(
            30, "--n-passes", help="Number of stochastic forward passes."
        ),
    ) -> None:
        """Estimate MC-Dropout uncertainty for a chest X-ray image."""
        raise NotImplementedError

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
