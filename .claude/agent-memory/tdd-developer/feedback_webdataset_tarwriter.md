---
name: feedback-webdataset-tarwriter
description: Use wds.TarWriter not wds.ShardWriter when writing a single predetermined shard path
metadata:
  type: feedback
---

Use `wds.TarWriter(shard_path)` instead of `wds.ShardWriter(shard_path, ...)` when you already know the exact output tar path.

**Why:** `ShardWriter` treats its first argument as a `%`-style format string and calls `pattern % self.shard` internally. Any path without a `%d`-style placeholder raises `TypeError: not all arguments converted during string formatting`. When the caller controls grouping and index offsets manually, `TarWriter` is the correct primitive.

**How to apply:** Whenever a shard path is already fully computed before the write loop, use `wds.TarWriter(shard_path)` as a context manager. Reserve `ShardWriter` only for cases where the library should manage shard rotation and naming via a `%d`-style pattern.
