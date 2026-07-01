# Memory Index — tdd-developer agent

- [Numpy bool assertions](feedback_numpy_bool_assertions.md) — use `== True/False` not `is True/False` for pandas/numpy cells from Parquet/DataFrame
- [Output dir creation](feedback_pipeline_dir_creation.md) — functions that write to caller-supplied paths must mkdir before writing; fsspec local does not auto-create parents
- [WebDataset TarWriter](feedback_webdataset_tarwriter.md) — use `wds.TarWriter` not `wds.ShardWriter` for predetermined shard paths
- [WebDataset .cls extension](feedback_webdataset_cls_extension.md) — `.cls` files with string labels must skip `.decode()`; handle raw bytes in `.map()` manually
- [DataLoader worker pickling](feedback_dataloader_worker_pickling.md) — Dataset classes for `num_workers > 0` must be module-level, not defined inside test functions
- [importlib.reload side effects](feedback_importlib_reload_side_effects.md) — use `patch.object(mod, "dep", None)` not reload+patch.dict to test absent optional deps
- [Lightning checkpoint weights_only](feedback_lightning_ckpt_weights_only.md) — `load_from_checkpoint` requires `weights_only=False` in PyTorch 2.6+ due to `functools.partial` in hparams
- [ONNX export patterns](feedback_onnx_export_patterns.md) — pass dummy input as tuple `(x,)`; MCD export: freeze net then re-enable Dropout layers only
- [Write files via Python](feedback_write_files_via_python.md) — use `python3 -c` or Write tool for files containing `!=` or shell-special chars
- [ETL implementation patterns](project_etl_implementation.md) — scikit-image import scope, functools.partial pickling, fsspec normalization, Prefect import guard, ParquetWriter guard, workers sentinel
- [Pipeline architecture](project_pipeline_architecture.md) — ops.py/prefect.py split pattern, compute_run_id hashing contract, Prefect 3 API facts
- [Manual ckpt fixture missing hook fields](feedback_manual_ckpt_fixture_missing_hook_fields.md) — hand-built checkpoint fixtures lack fields Lightning hooks add; check before relying on them
- [W&B sandbox env vars](feedback_wandb_sandbox_env_vars.md) — set WANDB_DIR/WANDB_DATA_DIR/WANDB_CACHE_DIR/WANDB_CONFIG_DIR to a writable tmp before running tests that build real `wandb.Artifact` objects

See also: [[shared/MEMORY.md]] for cross-agent rules (TDD, gh CLI, uv venv).
