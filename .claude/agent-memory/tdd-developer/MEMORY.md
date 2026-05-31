# Memory Index — tdd-developer agent

- [Numpy bool assertions](feedback_numpy_bool_assertions.md) — use `== True/False` not `is True/False` for pandas/numpy cells from Parquet/DataFrame
- [Output dir creation](feedback_pipeline_dir_creation.md) — functions that write to caller-supplied paths must mkdir before writing; fsspec local does not auto-create parents
- [WebDataset TarWriter](feedback_webdataset_tarwriter.md) — use `wds.TarWriter` not `wds.ShardWriter` for predetermined shard paths
- [Write files via Python](feedback_write_files_via_python.md) — use `python3 -c` or Write tool for files containing `!=` or shell-special chars
- [ETL implementation patterns](project_etl_implementation.md) — scikit-image import scope, functools.partial pickling, fsspec normalization, Prefect import guard, ParquetWriter guard, workers sentinel
- [Pipeline architecture](project_pipeline_architecture.md) — ops.py/prefect.py split pattern, compute_run_id hashing contract, Prefect 3 API facts

See also: [[shared/MEMORY.md]] for cross-agent rules (TDD, gh CLI, uv venv).
