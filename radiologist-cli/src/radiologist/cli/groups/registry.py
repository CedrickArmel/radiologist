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

"""``radiologist registry`` command group — Typer app fronting the W&B registry.

Commands: push, pull, resolve, promote, transition-to-production, list, and
a nested ``alias`` sub-app: get, set, remove. Grammar carried over verbatim
from the deleted ``radiologist-registry/src/radiologist/registry/cli.py``.
"""

from typing import List, Optional

import typer

app = typer.Typer(name="registry", add_completion=False)
alias_app = typer.Typer(name="alias", add_completion=False)
app.add_typer(alias_app, name="alias")

__all__ = ["app", "run"]


@app.command()
def push(
    det_path: str = typer.Option(
        ..., "--det-path", help="Path to the deterministic ONNX export."
    ),
    mcd_path: str = typer.Option(
        ..., "--mcd-path", help="Path to the MC-Dropout ONNX export."
    ),
    run_id: str = typer.Option(
        ..., "--run-id", help="W&B run ID to log the artifacts under."
    ),
    det_collection: str = typer.Option(
        ...,
        "--det-collection",
        help="Registry collection name for the deterministic artifact.",
    ),
    mcd_collection: str = typer.Option(
        ...,
        "--mcd-collection",
        help="Registry collection name for the MC-Dropout artifact.",
    ),
    input_shape: List[int] = typer.Option(
        ...,
        "--input-shape",
        help="Model input tensor shape, repeat once per dimension "
        "(e.g. --input-shape 1 --input-shape 3 --input-shape 224 "
        "--input-shape 224).",
    ),
    classes: List[str] = typer.Option(
        ...,
        "--classes",
        help="Ordered class labels the model predicts, repeatable "
        "(e.g. --classes NORMAL --classes PNEUMONIA).",
    ),
) -> None:
    """Open an ephemeral W&B run and log the deterministic and MC-Dropout artifacts."""
    raise NotImplementedError


@app.command()
def pull(
    path: str = typer.Argument(
        ...,
        help="Base artifact path, or a raw qualified artifact path when "
        "no selector flag is given.",
    ),
    local_dir: str = typer.Option(
        ..., "--local-dir", help="Directory to download the artifact into."
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Resolve the artifact logged by this run directly."
    ),
    tags: Optional[List[str]] = typer.Option(
        None, "--tags", help="Restrict the run search to these tag(s)."
    ),
    groups: Optional[List[str]] = typer.Option(
        None, "--groups", help="Restrict the run search to these group(s)."
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Summary metric used to rank candidate runs (highest first).",
    ),
    version: Optional[str] = typer.Option(
        None, "--version", help="Explicit version or alias to resolve."
    ),
    include_sweeps: bool = typer.Option(
        False,
        "--include-sweeps",
        help="Include sweep runs as eligible candidates.",
    ),
) -> None:
    """Resolve (if any selector flag is given) and download an ONNX artifact."""
    raise NotImplementedError


@app.command()
def resolve(
    path: str = typer.Argument(..., help="Base artifact path to resolve."),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Resolve the artifact logged by this run directly."
    ),
    tags: Optional[List[str]] = typer.Option(
        None, "--tags", help="Restrict the run search to these tag(s)."
    ),
    groups: Optional[List[str]] = typer.Option(
        None, "--groups", help="Restrict the run search to these group(s)."
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        help="Summary metric used to rank candidate runs (highest first).",
    ),
    version: Optional[str] = typer.Option(
        None, "--version", help="Explicit version or alias to resolve."
    ),
    include_sweeps: bool = typer.Option(
        False,
        "--include-sweeps",
        help="Include sweep runs as eligible candidates.",
    ),
) -> None:
    """Resolve a selector to a qualified artifact name and version."""
    raise NotImplementedError


@app.command()
def promote(
    path: str = typer.Argument(
        ..., help="Base artifact path shared by both artifacts."
    ),
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Run whose 'best' artifacts should be promoted.",
    ),
    det_collection: str = typer.Option(
        ...,
        "--det-collection",
        help="Collection to link the deterministic artifact to.",
    ),
    mcd_collection: str = typer.Option(
        ...,
        "--mcd-collection",
        help="Collection to link the MC-Dropout artifact to.",
    ),
    force: bool = typer.Option(False, "--force", help="Skip the confirmation prompt."),
) -> None:
    """Link both artifacts to their collections; prompts unless --force."""
    raise NotImplementedError


@app.command(name="transition-to-production")
def transition_to_production(
    det_collection: str = typer.Option(
        ...,
        "--det-collection",
        help="Collection holding the deterministic artifact.",
    ),
    mcd_collection: str = typer.Option(
        ...,
        "--mcd-collection",
        help="Collection holding the MC-Dropout artifact.",
    ),
    force: bool = typer.Option(False, "--force", help="Skip the confirmation prompt."),
) -> None:
    """Flip the 'staging' member of each collection to 'production'."""
    raise NotImplementedError


@app.command(name="list")
def list_(
    type_name: str = typer.Option(
        ..., "--type", help="Artifact type of the collection (e.g. 'model')."
    ),
    collection_name: str = typer.Option(
        ..., "--collection", help="Name of the collection to list."
    ),
) -> None:
    """List every member of a collection with its current aliases."""
    raise NotImplementedError


@alias_app.command("get")
def alias_get(
    artifact_path: str = typer.Argument(..., help="Fully qualified artifact path."),
) -> None:
    """Print the artifact's current aliases."""
    raise NotImplementedError


@alias_app.command("set")
def alias_set(
    artifact_path: str = typer.Argument(..., help="Fully qualified artifact path."),
    alias: str = typer.Argument(..., help="Alias to add."),
) -> None:
    """Add an alias to the artifact."""
    raise NotImplementedError


@alias_app.command("remove")
def alias_remove(
    artifact_path: str = typer.Argument(..., help="Fully qualified artifact path."),
    alias: str = typer.Argument(..., help="Alias to remove."),
) -> None:
    """Remove an alias from the artifact."""
    raise NotImplementedError


def run(argv: List[str]) -> int:
    """Run the ``registry`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``registry``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    raise NotImplementedError
