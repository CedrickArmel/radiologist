# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from typing import Callable, Dict, List, Optional

import lightning as L  # type: ignore[import-untyped]
import torch
import webdataset as wds  # type: ignore[import-untyped]

from radiologist.core.data.shards import (
    _count_samples,
    _discover_shards,
    _make_label_resolver,
)


class WebDatasetDataModule(L.LightningDataModule):
    """LightningDataModule that streams class-balanced batches from WebDataset tar shards.

    Args:
        shard_root: Root directory containing split/{label}/*.tar shards.
        label_map: Maps raw ETL label strings to class names.
        train_transform: Transform applied to training images (PIL → Tensor).
        eval_transform: Transform applied to val/test images (PIL → Tensor).
        train_loader: Partial factory for wds.WebLoader (training).
        eval_loader: Partial factory for wds.WebLoader (eval/test).
        classes: Ordered class names. If None, derived from sorted(set(label_map.values())).
        class_weights: Per-class mixing weights for RandomMix. Uniform when None.
        priors: Pre-computed class priors. Auto-computed from shard counts when None.
        shardshuffle: Buffer size for shard-level shuffle in train pipeline.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        shard_root: str,
        label_map: Dict[str, str],
        train_transform: Callable,
        eval_transform: Callable,
        train_loader: partial,
        eval_loader: partial,
        classes: Optional[List[str]] = None,
        class_weights: Optional[List[float]] = None,
        priors: Optional[List[float]] = None,
        shardshuffle: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.shard_root = shard_root
        self.label_map = label_map
        self.classes: List[str] = (
            classes if classes is not None else sorted(set(label_map.values()))
        )
        self.class_weights = class_weights
        self.priors = priors
        self.shardshuffle = shardshuffle
        self.seed = seed
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._shards: Optional[Dict[str, Dict[str, List[str]]]] = None

    @property
    def num_classes(self) -> int:
        """Number of classes; available immediately after construction."""
        return len(self.classes)

    def setup(self, stage: Optional[str] = None) -> None:
        """Discover shards and (optionally) compute priors.

        Args:
            stage: 'fit' discovers train+val splits; 'test' discovers test split.

        Raises:
            FileNotFoundError: If a required split has no shards.
            KeyError: If a discovered label directory is absent from label_map.
        """
        if stage == "test":
            splits = ["test"]
        else:
            splits = ["train", "val"]

        self._shards = _discover_shards(self.shard_root, splits)

        for split, by_label in self._shards.items():
            for label in by_label:
                if label not in self.label_map:
                    raise KeyError(
                        f"Label '{label}' found in shards but not in label_map"
                    )

        if self.priors is None and "train" in (self._shards or {}):
            self.priors = self._compute_priors()

    def _compute_priors(self) -> List[float]:
        assert self._shards is not None
        train_shards = self._shards.get("train", {})
        counts: Dict[str, int] = {cls: 0 for cls in self.classes}
        for label, paths in train_shards.items():
            cls_name = self.label_map[label]
            for p in paths:
                counts[cls_name] += _count_samples(p)
        total = sum(counts.values())
        return [counts[cls] / total for cls in self.classes]

    def _make_sample_mapper(self, transform: Callable) -> Callable:
        import io as _io

        from PIL import Image  # type: ignore[import-untyped]

        resolve = _make_label_resolver(self.label_map, self.classes)

        def decode_sample(sample: dict) -> dict:
            img = Image.open(_io.BytesIO(sample["png"])).convert("RGB")
            tensor = transform(img)
            label_str = sample["cls"].decode("utf-8").strip()
            target = torch.tensor(resolve(label_str), dtype=torch.int64)
            return {"input": tensor, "target": target, "key": sample["__key__"]}

        return decode_sample

    def _build_pipeline(
        self, shards_by_label: Dict[str, List[str]], transform: Callable, shuffle: bool
    ) -> wds.WebDataset:
        all_shards: List[str] = []
        for paths in shards_by_label.values():
            all_shards.extend(paths)

        ds = (
            wds.WebDataset(
                all_shards,
                shardshuffle=self.shardshuffle if shuffle else False,
            )
            .compose(wds.split_by_node, wds.split_by_worker)
            .map(self._make_sample_mapper(transform))
        )
        return ds

    def _build_class_pipelines(
        self, shards_by_label: Dict[str, List[str]], transform: Callable
    ) -> List[wds.WebDataset]:
        pipelines: Dict[int, wds.WebDataset] = {}

        for label, paths in shards_by_label.items():
            cls_name = self.label_map[label]
            cls_idx = self.classes.index(cls_name)

            ds = (
                wds.WebDataset(paths, shardshuffle=self.shardshuffle)
                .compose(wds.split_by_node, wds.split_by_worker)
                .map(self._make_sample_mapper(transform))
            )
            if cls_idx not in pipelines:
                pipelines[cls_idx] = ds
            else:
                existing = pipelines[cls_idx]
                additional = ds
                pipelines[cls_idx] = wds.RandomMix([existing, additional])  # type: ignore

        return [pipelines[i] for i in sorted(pipelines.keys())]

    def train_dataloader(self) -> wds.WebLoader:
        """Build class-balanced training loader via wds.RandomMix."""
        assert self._shards is not None, "Call setup('fit') before train_dataloader()"
        class_pipelines = self._build_class_pipelines(
            self._shards["train"], self.train_transform
        )
        n = len(class_pipelines)
        weights = (
            self.class_weights if self.class_weights is not None else [1.0 / n] * n
        )
        mixed = wds.RandomMix(class_pipelines, weights)
        return self._train_loader(mixed)

    def val_dataloader(self) -> wds.WebLoader:
        """Build deterministic val loader over all val shards."""
        assert self._shards is not None, "Call setup('fit') before val_dataloader()"
        pipeline = self._build_pipeline(
            self._shards["val"], self.eval_transform, shuffle=False
        )
        return self._eval_loader(pipeline)

    def test_dataloader(self) -> wds.WebLoader:
        """Build deterministic test loader over all test shards."""
        assert self._shards is not None, "Call setup('test') before test_dataloader()"
        pipeline = self._build_pipeline(
            self._shards["test"], self.eval_transform, shuffle=False
        )
        return self._eval_loader(pipeline)
