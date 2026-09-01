---
name: etl-three-stage-redesign
description: Fixed decisions for the radiologist-etl redesign — three stages (extract/assign-split/build), split-stability as an ML-correctness invariant, runner selection scoped to extract+build only
metadata:
  type: project
---

Decided with the user before spec work on **2026-09-01**; treat as settled requirements, not open
questions, when anything touches `radiologist-etl`:

- The monolithic `etl_flow` is replaced by **three independently invocable stages**, each with its
  own content-addressed `run_id`: **extract** (input = a *file listing image URIs*, never a
  directory scan; output = one batch manifest accumulating side-by-side in a shared folder),
  **assign-split** (input = a *folder* of extract manifests; output = one split-manifest), **build**
  (input = one split-manifest; output = shards under a folder named after its own run_id).
- **Split stability is an ML-correctness invariant, not a nicety.** A filename's train/val/test
  assignment must be a pure function of the filename plus an *ordered* ratio spec. If a filename
  could flip as the corpus grows, incremental runs leak train data into test and invalidate every
  eval number. This is why `split_ratios` moves from a YAML mapping to an ordered list of pairs.
- Runner/backend selection applies **only to extract and build**. Assign-split is deliberately
  local-only and out of scope for it.
- Three backend families are wanted long-term — Dask, Ray, **Apache Beam**. Beam is the constraint
  that shapes the design: it is *not* a Prefect `TaskRunner`, so any design built on
  `prefect_dask.DaskTaskRunner` makes Beam a permanent special case.
- Infrastructure provisioning (GKE Dask, KubeRay, Dataflow project/bucket, Flink) is **always out
  of scope** — ship the code-level abstraction and the config schema; addresses, cluster classes
  and GCP settings are values the operator supplies.
- The split-manifest row shape is a **cross-package contract**: `radiologist-core`'s datamodule
  imports `records_reader` from `radiologist.etl` and consumes a single split-manifest file. Any
  ETL redesign must preserve it so that package needs no edit.

**Why:** the user reached these through extended discussion and handed them over as fixed inputs to
architecture work, alongside a standing audit list (dropped `storage_options`, ratio-order
dependence, N+1 `fs.info` in `compute_run_id`, duplicated worker defaults, `records_reader`'s
required-positional `storage_options`, `@task`-wrapped functions in `__all__`, unchunked
per-image futures, silently dropped failed images).

**How to apply:** do not re-litigate the stage boundaries or re-propose a directory-scanning
extract. When designing for parallel execution here, check whether the proposal keeps Beam a peer
before committing to it. See also [[unified-cli-centralization]] (the ETL package stays a pure
library; commands live in `radiologist-cli`) and [[optional-feature-gating]] (dask/ray/beam are
third-party extras, which that rule permits — it bans feature flags on our own code) and
[[prefect-native-runner-selection]] (how the runner seam was actually shaped).
