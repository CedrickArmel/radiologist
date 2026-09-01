---
name: hydra-config-dir-group-override-quirks
description: Hydra CLI's --config-dir only adds group option files for defaults-list entries already written as "key: value" (a real config group); a bare "- trainer" defaults entry is a single-file include, not swappable via trainer=option, and each independently-defaulted subgroup (module/metric, module/loss, ...) re-merges after your override unless you also override or patch it.
metadata:
  type: feedback
---

Two Hydra CLI composition gotchas hit while building a tiny "test double"
config tree for `radiologist-cli`'s `core train` command (issue #175),
added via `--config-dir=<tmp_dir>` so CLI-argv-level tests could avoid the
production `module`/`datamodule` defaults (which point at real,
unavailable data):

1. **Bare defaults-list entries aren't swappable groups.** `train.yaml`
   has `- trainer` (no `: value`) in its `defaults:` list -- that's a
   plain single-file include (`trainer.yaml`, `@package _global_`), not a
   config group. `trainer=tiny` on the CLI fails with "Could not override
   'trainer'. No match in the defaults list." Only entries written as
   `- module: resnet50` / `- datamodule: default` (an explicit group +
   option) are swappable via `group=option`. For a bare entry, override
   its individual leaf keys instead: `trainer.max_epochs=1
   +trainer.limit_train_batches=2` (`+` prefix required for keys the
   base file doesn't already define).

2. **Independently-defaulted subgroups re-merge after your group swap.**
   `train.yaml`'s defaults list has `module: resnet50` AND separately
   `module/metric: fbeta_score`, `module/loss: focal_loss`,
   `module/scheduler: sequential`, `module/optimizer: adamw` -- each
   merges into the same `module` package independently, in defaults-list
   order. Overriding `module=tiny` does NOT stop `module/metric:
   fbeta_score` from still applying afterward and clobbering your tiny
   module's `metric` key (and it can carry interpolations like
   `${module.net.num_classes}` that break against your swapped-in net).
   Fix with a trailing CLI override that reasserts the value after all
   defaults-list merging: `module.metric.num_classes=2`.

**How to apply:** before assuming a `group=option` CLI override works,
check whether the target key appears in the defaults list as `key: value`
(swappable) or bare `key` (not swappable, use dotted/`+` overrides
instead). And enumerate every defaults-list entry sharing the same
`@package` as the group you're swapping -- each independent one still
applies on top of your swap.

See also [[hydra-cfg-passthrough-for-tests]], [[shared/MEMORY.md]].
