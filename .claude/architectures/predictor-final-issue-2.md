## ✨ Implement `Classifier` + `BasePredictor` core

### Context

This slice replaces the skeleton stubs for the classification path: `BasePredictor`'s two
constructors and shared helpers, plus `Classifier.predict`. It is the foundation every
other class builds on, so it lands first among the slices. Implement top-down through the
real helpers — reaching GREEN-real means no `NotImplementedError` is reachable through
`Classifier.from_path(...).predict(...)` or `Classifier.from_registry(...)`. See the epic
spec for the full architecture.

**Blocked by:** #1 (skeleton).

### User story

As a **caller doing chest-X-ray classification**, I want to **load a model and call
`predict`** so that **I get class probabilities and the predicted class without depending
on explanation or uncertainty machinery**.

### What to implement

**`base_predictor.py`** — port the monolith's session loading, metadata reading,
preprocessing, and prior correction:

- `BasePredictor.from_path(det_path, mcd_path=None)` — open `det_path` as an
  `ort.InferenceSession`, read custom metadata, optionally open `mcd_path`; store on
  internal state; return `cls.__new__(cls)`-built instance so subclasses get their own
  type. Keep the internal state container module-private.
- `BasePredictor.from_registry(artifact_path, local_dir, registry=None)` — when `registry`
  is `None`, instantiate `WandbRegistry()` from `radiologist-registry`; call
  `registry.pull(artifact_path, local_dir)` to get a local det path; delegate to
  `from_path`. The `registry` extra (wandb) absence must surface as a `RuntimeError`
  naming `registry` — rely on `WandbRegistry` raising it, or guard via the `_wandb`
  sentinel before instantiating, matching the existing pattern.

```python
@classmethod
def from_registry(
    cls,
    artifact_path: str,
    local_dir: str,
    registry: Optional["ModelRegistry"] = None,
) -> "BasePredictor":
    reg = registry if registry is not None else WandbRegistry()
    det_path = reg.pull(artifact_path=artifact_path, local_dir=local_dir)
    return cls.from_path(det_path=det_path)
```

- `_read_metadata(session)`, `_preprocess_image(image, input_shape)`,
  `_apply_prior_correction(softmax, classes, prior)` — port verbatim from the monolith
  (RGB conversion, bilinear resize to `(w, h)`, `/255`, CHW + batch axis; prior weighting
  + renormalization). These are module-private and exercised only through `predict`.

**`classifier.py`** — `Classifier.predict(image, deployment_prior=None)`:

- preprocess; run det session for `"logits"`; softmax in float64 with max-subtraction;
- effective prior = `deployment_prior`, else embedded `training_prior` from metadata when
  present, else none; apply correction when a prior exists;
- return `Prediction(probabilities={class: prob}, predicted_class=argmax)`.

Result dataclasses are imported from `models.py` (created in #1).

### Tests

Own the prediction behavioral tests (migrate the existing `test_predict.py` and the
`from_registry` cases from `test_registry.py`). Drive everything through `Classifier`'s
public API using the real ONNX fixtures (`build_det_onnx`, `det_onnx_path`); mock only the
wandb boundary (or inject a fake `registry` exposing `pull`).

- Given a det model and an image, `Classifier.from_path(...).predict(image)` returns a
  `Prediction` whose `probabilities` keys equal the model classes and whose
  `predicted_class` is the argmax of those probabilities.
- Given a `deployment_prior`, the returned probabilities differ from the no-prior result
  and remain a valid distribution (sum ≈ 1).
- Given a model with an embedded `training_prior` and no `deployment_prior`, the embedded
  prior is applied (result differs from a model without the embedded prior).
- `from_registry(artifact_path, local_dir, registry=<fake>)` returns a `Classifier` whose
  `predict` yields a `Prediction` with the same class keys as the equivalent `from_path`
  predictor (inject a fake `registry` whose `pull` returns the fixture path — no internal
  mocking).
- `from_registry` raises `RuntimeError` naming `registry` when the wandb extra is absent
  and no `registry` is injected.

### Acceptance criteria

- [ ] Given a det model and an image, `predict` returns a `Prediction` whose probability keys equal the model classes and whose `predicted_class` is the argmax.
- [ ] When a `deployment_prior` is supplied, the probabilities are prior-corrected and still sum to ≈ 1.
- [ ] When the model embeds a `training_prior` and no `deployment_prior` is given, the embedded prior is applied.
- [ ] `from_registry` with an injected registry returns a working predictor; with no registry and the wandb extra absent it raises `RuntimeError` naming `registry`.
- [ ] mypy clean; pytest green.
