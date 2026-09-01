---
name: prefect-native-runner-selection
description: For multi-backend execution in this repo, lean on Prefect's own TaskRunner + Hydra _target_ instantiation; never invent an ExecutionBackend interface
metadata:
  type: feedback
---

When a feature needs to run work on several execution backends (Dask, Ray, Beam, local), do
**not** design an abstract `ExecutionBackend`/`Runner` Protocol with one implementation per
backend. Prefect already defines the interface (`prefect.task_runners.TaskRunner`), ships
adapters (`prefect_dask.DaskTaskRunner`, `prefect_ray.RayTaskRunner`, and a built-in
`ProcessPoolTaskRunner` that covers the local CPU-bound case with zero extra deps), and Hydra
already instantiates classes from `_target_` — which this repo uses heavily in
`radiologist-core/configs`. So the whole abstraction is: a `runner/` Hydra config group whose
nodes carry a `_target_`, one resolution function, and `flow.with_options(task_runner=...)`.

Backends that genuinely are not task runners (Apache Beam owns its own parallelism and runs
inside a task body) are allowed to look structurally different — one concrete class, one
module, whose methods match the mapper callable shape the stage already accepts. Asymmetry
between backends is cheaper than a common interface nobody needs.

**Why:** the user's assigned strategy for the ETL redesign (2026-09-01) was explicitly
"TaskRunner-native — prioritize minimal new abstraction surface and reuse of Prefect-native
concepts over architectural symmetry". It is the same instinct as
[[no-protocol-null-object-ceremony]].

**How to apply:** inject parallelism into pure functions as a plain `Callable` type alias
(e.g. `BatchMapper = Callable[[Sequence[Sequence[str]]], List[BatchOutcome]]`) whose default is
a local pool — that keeps Prefect out of the library modules, keeps them testable with no flow
context, and is the same seam every backend plugs into. Note that mapped task calls are only
legal inside a flow run, so build the dispatching mapper in the flow and pass it down.
