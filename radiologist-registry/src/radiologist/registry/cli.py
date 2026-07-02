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

from typing import List, Optional

from radiologist.registry.optional import _TYPER_MISSING_MSG, _typer

if _typer is not None:
    import typer

    app = typer.Typer(name="radiologist-registry", add_completion=False)

    @app.command()
    def push(
        det_path: str = typer.Option(..., "--det-path"),
        mcd_path: str = typer.Option(..., "--mcd-path"),
        run_id: str = typer.Option(..., "--run-id"),
        det_collection: str = typer.Option(..., "--det-collection"),
        mcd_collection: str = typer.Option(..., "--mcd-collection"),
        input_shape: List[int] = typer.Option(..., "--input-shape"),
        classes: List[str] = typer.Option(..., "--classes"),
    ) -> None:
        raise NotImplementedError

    @app.command()
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
        raise NotImplementedError

    @app.command()
    def resolve(
        path: str = typer.Argument(...),
        run_id: Optional[str] = typer.Option(None, "--run-id"),
        tags: Optional[List[str]] = typer.Option(None, "--tags"),
        groups: Optional[List[str]] = typer.Option(None, "--groups"),
        metric: Optional[str] = typer.Option(None, "--metric"),
        version: Optional[str] = typer.Option(None, "--version"),
        include_sweeps: bool = typer.Option(False, "--include-sweeps"),
    ) -> None:
        raise NotImplementedError

    @app.command()
    def promote(
        path: str = typer.Argument(...),
        run_id: str = typer.Option(..., "--run-id"),
        det_collection: str = typer.Option(..., "--det-collection"),
        mcd_collection: str = typer.Option(..., "--mcd-collection"),
        force: bool = typer.Option(False, "--force"),
    ) -> None:
        raise NotImplementedError

    @app.command(name="transition-to-production")
    def transition_to_production(
        det_collection: str = typer.Option(..., "--det-collection"),
        mcd_collection: str = typer.Option(..., "--mcd-collection"),
        force: bool = typer.Option(False, "--force"),
    ) -> None:
        raise NotImplementedError

    @app.command(name="list")
    def list_(
        type_name: str = typer.Option(..., "--type"),
        collection_name: str = typer.Option(..., "--collection"),
    ) -> None:
        raise NotImplementedError

    alias_app = typer.Typer(name="alias", add_completion=False)
    app.add_typer(alias_app, name="alias")

    @alias_app.command("get")
    def alias_get(artifact_path: str = typer.Argument(...)) -> None:
        raise NotImplementedError

    @alias_app.command("set")
    def alias_set(
        artifact_path: str = typer.Argument(...), alias: str = typer.Argument(...)
    ) -> None:
        raise NotImplementedError

    @alias_app.command("remove")
    def alias_remove(
        artifact_path: str = typer.Argument(...), alias: str = typer.Argument(...)
    ) -> None:
        raise NotImplementedError

else:
    app = None  # type: ignore[assignment]


def main() -> None:
    if _typer is None:
        raise RuntimeError(_TYPER_MISSING_MSG)
    app()  # type: ignore[misc]
