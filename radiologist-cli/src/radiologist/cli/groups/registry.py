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

Bodies call straight into ``radiologist.registry``'s existing library
functions and render their result through :func:`radiologist.utils.cli.emit`
instead of ad-hoc ``typer.echo`` prose.

Error handling uses a module-local ``_exit_on_error`` decorator that mirrors
the intended contract of the still-stubbed
``radiologist.cli.errors.exit_on_error`` (owned by a sibling issue), built on
top of the already-implemented :func:`radiologist.utils.cli.exit_code_for`
taxonomy so exit codes are correct now and the decorator can be swapped for
the shared one later without changing behavior.
"""

import functools
from typing import Any, Callable, List, Optional, TypeVar

import typer

from radiologist.registry import ExportResult, WandbRegistry
from radiologist.registry import optional as _registry_optional
from radiologist.registry.selector import resolve_selector, selector_from_flags
from radiologist.utils.cli import emit, exit_code_for

app = typer.Typer(name="registry", add_completion=False)
alias_app = typer.Typer(name="alias", add_completion=False)
app.add_typer(alias_app, name="alias")

__all__ = ["app", "run"]

F = TypeVar("F", bound=Callable[..., None])


def _exit_on_error(func: F) -> F:
    """Wrap a command so an unhandled exception becomes a clean exit.

    Args:
        func: The command function to wrap.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except (typer.Abort, typer.Exit):
            raise
        except Exception as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=exit_code_for(exc))

    return wrapper  # type: ignore[return-value]


@app.command()
@_exit_on_error
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
    _registry_optional._guard_wandb()
    export_result = ExportResult(
        det_path=det_path,
        mcd_path=mcd_path,
        run_id=run_id,
        input_shape=tuple(input_shape),
        classes=classes,
    )
    run = _registry_optional._wandb.init(job_type="push")  # type: ignore[union-attr]
    logged = WandbRegistry().log_model_artifacts(
        export_result, run=run, ckpt_path=det_path
    )
    run.finish()
    emit(
        {
            "det_qualified_name": logged.det_qualified_name,
            "mcd_qualified_name": logged.mcd_qualified_name,
            "run_id": logged.run_id,
        }
    )


@app.command()
@_exit_on_error
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
    registry = WandbRegistry()
    selector = selector_from_flags(
        path=path,
        run_id=run_id,
        groups=groups,
        tags=tags,
        metric=metric,
        version=version,
        include_sweeps=include_sweeps,
    )
    if selector.is_registry_backed():
        ref = resolve_selector(selector, registry)
        local_path = registry.download(ref, local_dir)
    else:
        local_path = registry.pull(path, local_dir)
    emit({"local_path": local_path})


@app.command()
@_exit_on_error
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
    selector = selector_from_flags(
        path=path,
        run_id=run_id,
        groups=groups,
        tags=tags,
        metric=metric,
        version=version,
        include_sweeps=include_sweeps,
    )
    ref = resolve_selector(selector, WandbRegistry())
    emit({"qualified_name": ref.qualified_name, "version": ref.version})


@app.command()
@_exit_on_error
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
    if not force and not typer.confirm(
        f"Promote {path!r} (run {run_id!r}) to "
        f"{det_collection!r}/{mcd_collection!r}?"
    ):
        raise typer.Abort()
    result = WandbRegistry().promote(path, run_id, det_collection, mcd_collection)
    emit(
        {
            "det_qualified_name": result.det_qualified_name,
            "mcd_qualified_name": result.mcd_qualified_name,
            "alias": result.alias,
        }
    )


@app.command(name="transition-to-production")
@_exit_on_error
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
    if not force and not typer.confirm(
        f"Transition {det_collection!r}/{mcd_collection!r} to production?"
    ):
        raise typer.Abort()
    result = WandbRegistry().transition_to_production(det_collection, mcd_collection)
    emit(
        {
            "det_qualified_name": result.det_qualified_name,
            "mcd_qualified_name": result.mcd_qualified_name,
            "alias": result.alias,
        }
    )


@app.command(name="list")
@_exit_on_error
def list_(
    type_name: str = typer.Option(
        ..., "--type", help="Artifact type of the collection (e.g. 'model')."
    ),
    collection_name: str = typer.Option(
        ..., "--collection", help="Name of the collection to list."
    ),
) -> None:
    """List every member of a collection with its current aliases."""
    members = WandbRegistry().list_collection_artifacts(type_name, collection_name)
    for member in members:
        emit({"qualified_name": member.qualified_name, "aliases": member.aliases})


@alias_app.command("get")
@_exit_on_error
def alias_get(
    artifact_path: str = typer.Argument(..., help="Fully qualified artifact path."),
) -> None:
    """Print the artifact's current aliases."""
    aliases = WandbRegistry().get_aliases(artifact_path)
    emit({"artifact_path": artifact_path, "aliases": aliases})


@alias_app.command("set")
@_exit_on_error
def alias_set(
    artifact_path: str = typer.Argument(..., help="Fully qualified artifact path."),
    alias: str = typer.Argument(..., help="Alias to add."),
) -> None:
    """Add an alias to the artifact."""
    WandbRegistry().set_alias(artifact_path, alias)
    emit({"artifact_path": artifact_path, "alias": alias})


@alias_app.command("remove")
@_exit_on_error
def alias_remove(
    artifact_path: str = typer.Argument(..., help="Fully qualified artifact path."),
    alias: str = typer.Argument(..., help="Alias to remove."),
) -> None:
    """Remove an alias from the artifact."""
    WandbRegistry().remove_alias(artifact_path, alias)
    emit({"artifact_path": artifact_path, "alias": alias})


def run(argv: List[str]) -> int:
    """Run the ``registry`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``registry``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    try:
        result = app(args=argv, standalone_mode=False)
    except typer.Abort:
        return 1
    return result if isinstance(result, int) else 0
