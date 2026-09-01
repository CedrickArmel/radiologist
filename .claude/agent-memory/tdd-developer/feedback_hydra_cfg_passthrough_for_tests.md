---
name: hydra-cfg-passthrough-for-tests
description: hydra.main's decorated function accepts an optional cfg_passthrough positional arg that bypasses argv/CLI parsing entirely and calls the task function directly -- the cheapest way to unit-test a @hydra.main entry point's business logic with a real, hand-built DictConfig.
metadata:
  type: feedback
---

`hydra.main()`'s `decorated_main(cfg_passthrough=None)` wrapper: if called
with a `DictConfig` positional argument, it calls the wrapped task function
directly with that config and returns its result -- it never touches
`sys.argv`, never composes via the real config package, and is not a mock
(it's the documented, first-class calling convention `hydra.main` itself
defines).

**Why:** discovered implementing `radiologist-cli`'s `core train` command
(issue #175). Production Hydra config groups (`module: resnet50`,
`datamodule: default`) point at real, unavailable data/models, so composing
the CLI's actual `train.yaml` via the argv path is unusable for most
business-logic tests. Calling `train_main(cfg)` directly with a hand-built
`DictConfig` (tiny real net, tiny real WebDataset shards written to
`tmp_path`) exercises the exact same function body with real Lightning
components -- no `radiologist.*` mocking -- while staying fast.

**How to apply:** for any `@hydra.main`-decorated entry point, split tests
into two tiers:
1. Business-logic / schema / return-value tests -> call the decorated
   function directly with a manually composed `DictConfig` (cfg
   passthrough). Fast, no data dependency, no argv/subprocess needed.
2. CLI-surface tests (`--help`, `key=value` overrides, group overrides,
   exit codes) -> exercise the real argv path, via `subprocess` + the
   project's own multi-package `PYTHONPATH` shim (mirroring the root
   `conftest.py`). When production config groups point at real
   unavailable data, add a throwaway Hydra `--config-dir` with tiny
   `module`/`datamodule` group yaml files (see
   [[hydra-config-dir-group-override-quirks]]) rather than touching the
   production config package.

See also [[hydra-full-error-for-exit-codes]], [[shared/MEMORY.md]].
