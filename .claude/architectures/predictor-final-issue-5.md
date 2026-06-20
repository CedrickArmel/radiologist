## ✨ Refactor `app.py` smart factory + `cli.py` three subcommands

### Context

With all three concrete predictor classes implemented, this slice replaces the skeleton
`create_app` stub with a smart `isinstance`-driven factory and rewrites the CLI into three
focused subcommands. Reaching GREEN-real means `create_app` wires real routes per
predictor type and the CLI commands invoke the real classes — no `NotImplementedError`
reachable through either entry point. See the epic spec for the route-matching contract.

**Blocked by:** #1 (skeleton), #2 (`Classifier`), #3 (`Explainer`), #4
(`MCDropoutPredictor`) — the factory and CLI must wire all three concrete classes.

### User story

As an **operator deploying a model**, I want **the HTTP server and CLI to expose exactly
the capabilities the loaded model supports** so that **I never get a route or command that
silently fails because the model can't serve it**.

### What to implement

**`app.py`** — `_build_app` becomes capability-aware and `create_app` selects routes via
`isinstance`:

- always wire `GET /healthz`;
- wire `POST /predict` when the predictor is a `Classifier` (covers `Explainer` too, since
  it inherits `Classifier`);
- wire `POST /explain` when the predictor is an `Explainer`;
- wire `POST /uncertainty` when the predictor is an `MCDropoutPredictor`;
- a route the instance cannot serve must be **absent** (FastAPI returns 404), not a 503/501
  stub.
- keep the `serve` extra guard: `create_app` raises `RuntimeError` naming `serve` when
  fastapi is absent.

```python
def create_app(predictor: Optional["BasePredictor"] = None) -> Any:
    if _fastapi is None:
        raise RuntimeError("The 'serve' extra is required.")
    return _build_app(_fastapi, predictor)
```

`_build_app` reads the predictor type once and registers only the matching route handlers.
Reuse the existing image-loading / validation helpers and the 503-when-no-predictor guard
for routes that *are* wired but have no predictor injected.

**`cli.py`** — three subcommands, no `pull`:

- `predict <image_path> --model <det_path> [--prior ...]` → `Classifier.from_path(...)`,
  print predicted class + probabilities;
- `explain <image_path> --model <det_path> [--out <path>]` → `Explainer.from_path(...)`,
  produce the saliency map (echo summary / save when `--out` given);
- `uncertainty <image_path> --model <det_path> --mcd-model <mcd_path> [--n-passes 30]` →
  `MCDropoutPredictor.from_path(det_path, mcd_path=...)`, print mean/std/entropy.
- keep the `_typer is None` guard and `main()` raising `RuntimeError` naming the `cli`
  extra when typer is absent. No `pull` subcommand — that lives in the registry package.

### Tests

Own the serving and CLI behavioral tests (migrate `test_app.py`, `test_cli.py`). Drive
through `create_app` (via a FastAPI `TestClient`) and the Typer `CliRunner`, using real
ONNX fixtures; mock only true boundaries.

- `create_app(Classifier(...))` serves `POST /predict` (200 with probabilities) and
  `GET /healthz`, and returns 404 for `POST /explain` and `POST /uncertainty`.
- `create_app(Explainer(...))` additionally serves `POST /explain` (200 with a saliency
  map) while still serving `/predict`.
- `create_app(MCDropoutPredictor(...))` serves `POST /uncertainty` (200 with mean/std/
  entropy) and `GET /healthz`, and 404s on `/predict`.
- A wired route with no predictor injected returns 503; `GET /healthz` reflects the same.
- CLI `predict` on a det model prints the predicted class and exits 0; `explain` produces a
  saliency result and exits 0; `uncertainty` with a det+mcd model prints uncertainty stats
  and exits 0.
- The CLI exposes no `pull` subcommand (invoking `pull` errors as unknown command).

### Acceptance criteria

- [ ] `create_app(Classifier(...))` serves `/predict` and `/healthz` and 404s on `/explain` and `/uncertainty`.
- [ ] `create_app(Explainer(...))` additionally serves `/explain`; `create_app(MCDropoutPredictor(...))` serves `/uncertainty` and 404s on `/predict`.
- [ ] A wired route with no predictor injected returns 503.
- [ ] The CLI exposes `predict`, `explain`, and `uncertainty` subcommands that run the corresponding classes and exit 0 on success; there is no `pull` subcommand.
- [ ] `create_app` raises `RuntimeError` naming `serve` when fastapi is absent; `main()` raises `RuntimeError` naming `cli` when typer is absent.
- [ ] mypy clean; pytest green.
