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

"""Typer-based CLI for radiologist-registry.

Entry points: push, pull, resolve, promote, transition-to-production, list,
              alias get/set/remove
"""

import functools
from typing import Any, Callable, List, Optional, TypeVar

from radiologist.registry.models import ExportResult
from radiologist.registry.optional import _wandb  # noqa: F401
from radiologist.registry.optional import _TYPER_MISSING_MSG, _guard_wandb, _typer
from radiologist.registry.selector import RegistrySelector, resolve_selector
from radiologist.registry.wandb_registry import WandbRegistry

F = TypeVar("F", bound=Callable[..., None])

if _typer is not None:
    import typer

    app = typer.Typer(name="radiologist-registry", add_completion=False)

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
    def push(
        det_path: str = typer.Option(..., "--det-path"),
        mcd_path: str = typer.Option(..., "--mcd-path"),
        run_id: str = typer.Option(..., "--run-id"),
        det_collection: str = typer.Option(..., "--det-collection"),
        mcd_collection: str = typer.Option(..., "--mcd-collection"),
        input_shape: List[int] = typer.Option(..., "--input-shape"),
        classes: List[str] = typer.Option(..., "--classes"),
    ) -> None:
        _guard_wandb()
        export_result = ExportResult(
            det_path=det_path,
            mcd_path=mcd_path,
            run_id=run_id,
            input_shape=tuple(input_shape),
            classes=classes,
        )
        run = _wandb.init(job_type="push")  # type: ignore[union-attr]
        logged = WandbRegistry().log_model_artifacts(
            export_result, run=run, ckpt_path=det_path
        )
        run.finish()
        typer.echo(logged.det_qualified_name)
        typer.echo(logged.mcd_qualified_name)

    @app.command()
    @_exit_on_error
    def pull(
        path: str = typer.Argument(...),
        local_dir: str = typer.Option(..., "--local-dir"),
        run_id: Optional[str] = typer.Option(None, "--run-id"),
        tags: Optional[List[str]] = typer.Option(None, "--tags"),
        groups: Optional[List[str]] = typer.Option(None, "--groups"),
        metric: Optional[str] = typer.Option(None, "--metric"),
        version: Optional[str] = typer.Option(None, "--version"),
        include_sweeps: bool = typer.Option(False, "--include-sweeps"),
    ) -> None:
        registry = WandbRegistry()
        selector = RegistrySelector(
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
        typer.echo(local_path)

    @app.command()
    @_exit_on_error
    def resolve(
        path: str = typer.Argument(...),
        run_id: Optional[str] = typer.Option(None, "--run-id"),
        tags: Optional[List[str]] = typer.Option(None, "--tags"),
        groups: Optional[List[str]] = typer.Option(None, "--groups"),
        metric: Optional[str] = typer.Option(None, "--metric"),
        version: Optional[str] = typer.Option(None, "--version"),
        include_sweeps: bool = typer.Option(False, "--include-sweeps"),
    ) -> None:
        selector = RegistrySelector(
            path=path,
            run_id=run_id,
            groups=groups,
            tags=tags,
            metric=metric,
            version=version,
            include_sweeps=include_sweeps,
        )
        ref = resolve_selector(selector, WandbRegistry())
        typer.echo(ref.qualified_name)
        typer.echo(ref.version)

    @app.command()
    @_exit_on_error
    def promote(
        path: str = typer.Argument(...),
        run_id: str = typer.Option(..., "--run-id"),
        det_collection: str = typer.Option(..., "--det-collection"),
        mcd_collection: str = typer.Option(..., "--mcd-collection"),
        force: bool = typer.Option(False, "--force"),
    ) -> None:
        if not force and not typer.confirm(
            f"Promote {path!r} (run {run_id!r}) to "
            f"{det_collection!r}/{mcd_collection!r}?"
        ):
            raise typer.Abort()
        result = WandbRegistry().promote(path, run_id, det_collection, mcd_collection)
        typer.echo(f"{result.det_qualified_name} -> {result.alias}")
        typer.echo(f"{result.mcd_qualified_name} -> {result.alias}")

    @app.command(name="transition-to-production")
    @_exit_on_error
    def transition_to_production(
        det_collection: str = typer.Option(..., "--det-collection"),
        mcd_collection: str = typer.Option(..., "--mcd-collection"),
        force: bool = typer.Option(False, "--force"),
    ) -> None:
        if not force and not typer.confirm(
            f"Transition {det_collection!r}/{mcd_collection!r} to production?"
        ):
            raise typer.Abort()
        result = WandbRegistry().transition_to_production(
            det_collection, mcd_collection
        )
        typer.echo(f"{result.det_qualified_name} -> {result.alias}")
        typer.echo(f"{result.mcd_qualified_name} -> {result.alias}")

    @app.command(name="list")
    @_exit_on_error
    def list_(
        type_name: str = typer.Option(..., "--type"),
        collection_name: str = typer.Option(..., "--collection"),
    ) -> None:
        members = WandbRegistry().list_collection_artifacts(type_name, collection_name)
        for member in members:
            typer.echo(f"{member.qualified_name} {member.aliases}")

    alias_app = typer.Typer(name="alias", add_completion=False)
    app.add_typer(alias_app, name="alias")

    @alias_app.command("get")
    @_exit_on_error
    def alias_get(artifact_path: str = typer.Argument(...)) -> None:
        aliases = WandbRegistry().get_aliases(artifact_path)
        typer.echo(" ".join(aliases))

    @alias_app.command("set")
    @_exit_on_error
    def alias_set(
        artifact_path: str = typer.Argument(...), alias: str = typer.Argument(...)
    ) -> None:
        WandbRegistry().set_alias(artifact_path, alias)
        typer.echo(f"Set alias {alias!r} on {artifact_path!r}")

    @alias_app.command("remove")
    @_exit_on_error
    def alias_remove(
        artifact_path: str = typer.Argument(...), alias: str = typer.Argument(...)
    ) -> None:
        WandbRegistry().remove_alias(artifact_path, alias)
        typer.echo(f"Removed alias {alias!r} from {artifact_path!r}")

else:
    app = None  # type: ignore[assignment]


def main() -> None:
    if _typer is None:
        raise RuntimeError(_TYPER_MISSING_MSG)
    app()  # type: ignore[misc]
