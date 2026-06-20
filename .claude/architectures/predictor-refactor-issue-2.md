## ✨ Preprocessing + metadata seam

### Context

This is the shared seam-issue: `preprocess_image` and `read_metadata` are consumed by every capability slice (`Classifier`, `Explainer`, `MCDropoutPredictor`) and by `BasePredictor.from_path`. Because more than one slice depends on them, they are promoted to their own issue with behavioral contracts at the module boundary; the capability slices `Requires` this. This slice replaces the skeleton stubs in `preprocessing.py` with the real logic ported verbatim from the legacy `predictor.py` private helpers `_preprocess_image` / `_read_metadata`. Also replaces the `BasePredictor.metadata` property and `BasePredictor.from_path` stubs so subclasses inherit a working session lifecycle. Requires: #1. Target GREEN-real: no `NotImplementedError` reachable through `preprocess_image`, `read_metadata`, `BasePredictor.from_path`, or `BasePredictor.metadata`.

### User story

As an **inference capability author**, I want a single tested preprocessing + metadata seam so that every capability reads images and model metadata identically without duplicating the logic.

### Acceptance criteria

- [ ] Given an RGB image path and an `[N, C, H, W]` input shape, the preprocessing seam returns a float32 array of shape `(1, C, H, W)` with all values in `[0, 1]`.
- [ ] Given an HWC uint8 numpy array, the seam returns the same `(1, C, H, W)` float32 shape, resized to the requested `H`/`W`.
- [ ] Given a PIL image in a non-RGB mode, the seam converts to RGB before normalizing (output has `C == 3`).
- [ ] Given a model loaded from a valid ONNX path, the typed metadata view exposes the embedded `classes`, `input_shape`, and `cam_target_layer`.
- [ ] When `from_path` is given a non-existent path, it raises `FileNotFoundError`.
- [ ] mypy clean; pytest green.

### Technical notes

- Port `_preprocess_image` (legacy `predictor.py:86-110`) and `_read_metadata` (`predictor.py:81-83`) verbatim; only the names lose the underscore and the home module changes.
- `BasePredictor.from_path` opens `ort.InferenceSession(model_path)` and stores the session + `read_metadata(session)` result on the instance. Use `cls.__new__(cls)` + attribute assignment (matches the legacy construction at `predictor.py:179-185`) so the ABC stays instantiable by subclasses.
- `metadata` property parses the stored string-map into `ModelMetadata`; `mc_dropout` defaults to `False` when the key is absent, `output_names` JSON-decodes from the `output_names` key.
- Do not test `preprocess_image` by mocking — feed it the real fixture images / arrays from `conftest.py`.

### Design notes

`BasePredictor` is an ABC, not a Protocol: `from_path`, `from_registry`, and metadata parsing are real shared implementation that every subclass inherits unchanged. A Protocol would declare structure but carry no code, forcing each capability to re-implement loading — the opposite of the reuse this decomposition exists to achieve. The ONNX session lives on the base; subclasses only add capability methods that read `self`'s session and metadata.
