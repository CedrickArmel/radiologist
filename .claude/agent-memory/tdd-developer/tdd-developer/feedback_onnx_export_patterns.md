---
name: onnx-export-patterns
description: Patterns for safe torch.onnx.export calls and ONNX metadata_props in this project
metadata:
  type: feedback
---

Pass input args as a tuple to `torch.onnx.export`: use `(dummy_input,)` not `dummy_input` to satisfy mypy's `tuple[Any, ...]` type constraint.

For MC-dropout export: call `model.train(mode=False)` on the full net first, then re-enable only `nn.Dropout` layers with `for m in net.modules(): if isinstance(m, nn.Dropout): m.train()`. Export with `training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=False`. Requires `onnxscript` installed alongside `onnx`.

**Why:** The deterministic/stochastic distinction requires BatchNorm frozen (running stats) and Dropout active only for MCD.

**How to apply:** Same pattern for any future ONNX export that needs MC-dropout inference.
