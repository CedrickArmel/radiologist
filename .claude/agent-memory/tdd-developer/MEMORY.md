# Memory Index — tdd-developer agent

- [ETL retire legacy flow #190](project_etl_retire_legacy_flow_190.md) — final epic refactor: ops.py/etl_flow/*_task/EtlResult were all dead, deletion-only, zero test changes
- [ETL runner selection #182](project_etl_runner_selection_182.md) — ExecutionPlan/resolve_execution contract, conf/runner/*.yaml shape, mock-the-SDK-not-owned-code pattern for absent dask/ray/beam
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
- [pyenv activate needs shell init](feedback_pyenv_activate_needs_shell_init.md) — Bash calls need `eval "$(pyenv init -)"; eval "$(pyenv virtualenv-init -)"` before `pyenv activate`, else installs silently land in the shared venv
- [Worktree branch already checked out](feedback_worktree_branch_already_checked_out.md) — git refuses a 2nd worktree for a branch checked out elsewhere; work in-place instead; also watch for uv.lock revision noise from `make dev-install`
- [Issue test-scope incomplete](feedback_issue_test_scope_incomplete.md) — an issue's test-update list can omit call sites broken by a signature change; grep the old signature across the whole suite before declaring green
- [onnxruntime NoSuchFile vs FileNotFoundError](feedback_onnxruntime_no_such_file_vs_file_not_found.md) — onnxruntime raises its own NoSuchFile on a missing path, not FileNotFoundError; check existence explicitly if the AC requires the stdlib type
- [Epic seam convention ownership move](feedback_epic_seam_convention_ownership_move.md) — when a new shared seam's contract says a convention "now lives here"/"is no longer done automatically", grep sibling methods for the old inline version and simplify it away, don't double-apply
- [Narrowed shared-helper signature drops params](feedback_narrowed_shared_helper_signature_drops_params.md) — a "rewire onto shared helper X" issue's literal call-signature snippet can predate optional kwargs that landed afterward; grep old call sites' full kwargs before deleting any, extend the helper instead of regressing
- [Review-fix reverts epic seam scope creep](feedback_review_fix_reverts_epic_seam_scope_creep.md) — when a review says an internal seam leaked into `__all__`, revert both the export and the test's expected set atomically; don't just widen the test to match the leak
- [Spawn pool test module PYTHONPATH](feedback_spawn_pool_test_module_pythonpath.md) — process-pool tests need PYTHONPATH set (not just sys.path) so spawned children can import the test module's worker function
- [OOM kills pytest under parallel agents](feedback_oom_kills_pytest_under_parallel_agents.md) — exit 137 + empty output looks like a hang/pass; run `-u`, file-by-file, and inject a serial mapper in your own tests
- [Docstring lint cutover scope](feedback_docstring_lint_cutover_scope.md) — a "clean" claim measured under a still-suppressed rule isn't proof the flip is clean; re-verify after; scope src-only enforcement via per-file-ignores instead of writing test docstrings

- [Epic slice readonly seam reuse](feedback_epic_slice_readonly_seam_reuse.md) — verify a "already implemented, check if it's a gap" hedge by reading the file before writing extra code
- [Third-party version string in AC](feedback_third_party_version_string_in_ac.md) — an issue's literal example of a dependency's output string can drift from the pinned lockfile version; assert against the library's own constant, not the literal
- [Worktree venv .pth corruption risk](feedback_worktree_venv_pth_corruption_risk.md) — `uv sync/run --active` needs both `VIRTUAL_ENV` and `PATH` exported in the same Bash call or it silently rewrites the shared venv's `.pth` files
- [Optional refactor won't-do bar](feedback_optional_refactor_wont_do_bar.md) — when an optional issue states its own "not an improvement" threshold, apply it literally and report won't-do
- [Publish workflow resolve pattern](project_publish_workflow_resolve_pattern.md) — checkout merge_commit_sha directly (it's known pre-checkout); tomli must be explicit release-group dep on py3.10
- [Worktree shell chaining blocked](feedback_worktree_shell_chaining_blocked.md) — worktree-isolated Bash sandbox rejects any `&&`/`;`/`$VAR` shell line; use single flat commands with an inline `VAR=val /abs/bin` prefix instead of `pyenv activate`/`eval`
- [cz uv provider cwd-relative lockfile](feedback_cz_uv_provider_cwd_relative_lockfile.md) — commitizen's `uv` version provider resolves `uv.lock`/`pyproject.toml` against cwd only; bumping a workspace member from its own dir needs a symlinked `uv.lock`
- [Worktree scratch-clone shares gitdir](feedback_worktree_scratch_clone_shares_gitdir.md) — never `cp -r` a git worktree for a git sandbox; its `.git` is a pointer file to the shared repo, so commits/tags there pollute the real repo's refs
- [Prefect broken local server, use .fn bypass](feedback_prefect_broken_local_server_use_fn_bypass.md) — local ephemeral Prefect API is broken (Starlette mismatch), real Cloud creds present in env; use `.fn` + stub artifact calls to test real business logic
- [CLI run() env var must restore](feedback_cli_run_env_var_must_restore.md) — a group's `run(argv)` setting `RADIOLOGIST_OUTPUT` for a `@hydra.main` entry point must save/restore it in `finally`, or it leaks across the whole pytest session

- [ETL assign-split stage #184](project_etl_assign_split_184.md) — ordered-ratio contract, dedup/collision handling, ops.py call-site fix, pre-commit mypy checks tests too

See also: [[shared/MEMORY.md]] for cross-agent rules (TDD).
