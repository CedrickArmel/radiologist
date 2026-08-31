# Memory Index

- [Parallel output namespacing](feedback_parallel_output_namespacing.md) — in Mode A, prefix draft files by strategy to avoid clobbering sibling agents in shared .claude/architectures
- [Optional feature gating](feedback_optional_feature_gating.md) — no opt-in flags/new extras for in-package capabilities; extras for whole workspace members are fine
- [Unified CLI centralization](project_unified_cli_centralization.md) — milestone #15: all command code moves into radiologist-cli; the four business packages lose their console scripts
- [No Protocol/Null-Object ceremony](feedback_no_protocol_null_object_ceremony.md) — internal single-impl features get one concrete class in one module, not an interface + null pair + extra file
- [Per-application resource lifetime](feedback_per_application_resource_lifetime.md) — build stateful registries inside create_app(), never module-level; the suite builds one app per fixture
- [Pytest layout](project_pytest_layout.md) — importlib mode, repo-root rootdir, conftest discovery, src-shim pattern; root conftest must insert all 5 src/ dirs
