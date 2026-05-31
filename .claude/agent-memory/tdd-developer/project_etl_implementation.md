---
name: project-etl-implementation
description: radiologist-etl key constraints and gotchas for stats, processors, filters, manifest, shards
metadata:
  type: project
---

Package lives at `radiologist-etl/src/radiologist/etl/`. Tests in `radiologist-etl/tests/`.

**scikit-image import constraint:** all `from skimage.feature import ...` calls must stay inside function bodies in `stats.py` — never at module level. This keeps the import lazy and avoids circular issues.

**Multiprocessing pickling:** any callable submitted to a process pool must be picklable — use `functools.partial` of a top-level function; closures are not picklable on OSes that use `spawn` instead of `fork` (e.g. macOS).

**fsspec path normalization in `_resolve_mask`:** `fsspec.unstrip_protocol` on local paths returns `file:///path/...` URIs. Raw string prefix-stripping fails silently. Always normalize both sides through `fsspec.url_to_fs` before computing relative paths:
```python
_, norm_image = fsspec.url_to_fs(image_path, **opts)
_, norm_root = fsspec.url_to_fs(images_root, **opts)
rel = norm_image[len(norm_root):].lstrip("/")
```

**Prefect import guard + mypy:** Wrap all `from prefect import ...` in `try/except ImportError` with stub no-ops (in `prefect.py`). The stubs need `# type: ignore[misc, no-redef]` (not just `[misc]`) because mypy sees the try-branch import and the except-branch `def` as a redefinition.

**ParquetWriter empty guard:** `ParquetWriter.write` raises `ValueError("Cannot write an empty manifest — no records to persist.")` when called with an empty list. Never skip this guard — a schema-less Parquet breaks all downstream steps.

**StatsProcessor.workers sentinel:** `workers` parameter is `int | None = None`; resolved at runtime as `self._workers = workers or os.cpu_count() or 1`. Never use `os.cpu_count()` as a default argument value — it is evaluated once at import time.

**build_shards storage_options:** `build_shards` accepts `storage_options: dict | None = None` and forwards it to the underlying fsspec calls.
