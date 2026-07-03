# Memory Index — tdd-developer agent

- [Numpy bool assertions](feedback_numpy_bool_assertions.md) — use `== True/False` not `is True/False` for pandas/numpy cells from Parquet/DataFrame
- [Pre-commit E402](feedback_precommit_e402.md) — add `# noqa: E402` to imports that must follow a `sys.path.insert(...)` shim
- [Lazy imports for RED-phase tests](feedback_test_import.md) — import not-yet-implemented functions inside the test method body, not at module level, to avoid collection-time ImportError masking the real FAILED test
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
- [Skeleton issue repoints old tests](feedback_skeleton_issue_repoints_old_tests.md) — when a skeleton changes a CLI's command surface, update shape-only assertions in the existing test file rather than leaving it broken
- [Shared base-class seam in decomposition epics](feedback_shared_base_class_seam_in_decomposition_epics.md) — implement the minimal real shared-seam code your own GREEN-real bar needs, even if a sibling issue "owns" it
- [Testing optional import guards](feedback_testing_optional_import_guards.md) — sentinel-patch tests don't prove a try/except guard exists; use blocked-import + reload for the real red test
- [Hydra main test via subprocess](feedback_hydra_main_test_via_subprocess.md) — test a `@hydra.main` entry point's --help/exit-0 via subprocess, not in-process sys.argv monkeypatch
- [Typer CLI + wandb process boundary](feedback_typer_cli_wandb_process_boundary.md) — a CLI module's own `_wandb` sentinel and a facade submodule's `_wandb` are separate bindings; patch each independently
- [Registry download vs pull semantics](feedback_registry_download_vs_pull_semantics.md) — `.download()` is ckpt-only (training resume), `.pull()` is onnx-only (inference load); don't conflate even if a spec says "download"
- [Pytest worktree isolation](feedback_pytest_worktree_isolation.md) — stale `.pth` cross-worktree pointers and cross-package plugin-name collisions; run pytest per-package with `--confcutdir=.`

See also: [[shared/MEMORY.md]] for cross-agent rules (TDD).
