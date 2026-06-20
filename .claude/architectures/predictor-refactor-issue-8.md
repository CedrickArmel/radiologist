## ✨ Three-subcommand CLI (`predict` / `explain` / `uncertainty`)

### Context

Replaces the legacy two-command CLI (`predict`, `pull`) with three capability-aligned subcommands: `predict` (Classifier), `explain` (Explainer), and `uncertainty` (MCDropoutPredictor). The `pull` subcommand is removed entirely — model download is owned by the `radiologist-registry` package's own surface. Each subcommand constructs only the capability it needs from a single `--model` path. Requires: #3, #4, #5 (the capability methods each subcommand invokes must be GREEN-real). Target GREEN-real: no `NotImplementedError` reachable through any subcommand.

### User story

As a **command-line user**, I want one subcommand per capability so that I run only the model I need and there is no leftover registry command in the inference CLI.

### Acceptance criteria

- [ ] Invoking `predict <image> --model <det.onnx>` prints the predicted class and per-class probabilities and exits 0.
- [ ] Invoking `explain <image> --model <det.onnx>` prints the predicted class and reports the saliency map (e.g. its shape) and exits 0.
- [ ] Invoking `uncertainty <image> --model <mcd.onnx>` prints mean probabilities and predictive entropy and exits 0.
- [ ] Any subcommand given a bad model or image path prints an error to stderr and exits with a non-zero code.
- [ ] There is no `pull` subcommand.
- [ ] When the `cli` extra (typer) is not installed, invoking the CLI entry point raises `RuntimeError` naming the `cli` extra.
- [ ] mypy clean; pytest green.

### Technical notes

- `predict` builds `Classifier.from_path(model)`, `explain` builds `Explainer.from_path(model)`, `uncertainty` builds `MCDropoutPredictor.from_path(model)`. No `--mcd-model` flag — each capability takes a single `--model`.
- Keep the `_typer is None` guard and `main()` entry point from legacy `cli.py:34-90`; keep per-command try/except printing `Error: {exc}` to stderr and `raise typer.Exit(code=1)`.
- Remove the import of `Predictor` / `pull_model` from `cli.py`; import the capability classes instead.
- `test_cli.py` currently tests `predict` and `pull`; update it to the three-subcommand contract (the `pull` test is removed — that behavior now belongs to the registry package's CLI/tests). This is intended behavior change to the public CLI surface.
