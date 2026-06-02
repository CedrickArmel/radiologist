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

import pytest
import torch
import webdataset as wds  # type: ignore[import-untyped]

from radiologist.core import WebDatasetDataModule


def _make_dm(
    shard_root,
    label_map,
    classes,
    train_transform,
    eval_transform,
    train_loader_partial,
    eval_loader_partial,
    **kwargs,
) -> WebDatasetDataModule:
    return WebDatasetDataModule(
        shard_root=str(shard_root),
        label_map=label_map,
        train_transform=train_transform,
        eval_transform=eval_transform,
        train_loader=train_loader_partial,
        eval_loader=eval_loader_partial,
        classes=classes,
        **kwargs,
    )


class TestNumClasses:
    def test_num_classes_available_before_setup(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        assert dm.num_classes == len(classes)

    def test_num_classes_derived_from_label_map_when_classes_is_none(
        self,
        shard_root,
        label_map,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = WebDatasetDataModule(
            shard_root=str(shard_root),
            label_map=label_map,
            train_transform=train_transform,
            eval_transform=eval_transform,
            train_loader=train_loader_partial,
            eval_loader=eval_loader_partial,
        )
        expected = sorted(set(label_map.values()))
        assert dm.num_classes == len(expected)
        assert dm.classes == expected


class TestSetupFit:
    def test_setup_fit_succeeds_with_valid_shard_layout(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("fit")

    def test_setup_fit_raises_file_not_found_when_train_shards_missing(
        self,
        tmp_path,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        dm = _make_dm(
            empty_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        with pytest.raises(FileNotFoundError):
            dm.setup("fit")

    def test_setup_fit_raises_key_error_when_label_not_in_label_map(
        self,
        tmp_path,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        import io
        import tarfile

        root = tmp_path / "shards"
        for split in ("train", "val"):
            for label in ("NORMAL", "ABNORMAL", "UNKNOWN"):
                shard_path = (
                    root / split / label / f"{split}-{label.lower()}-000000.tar"
                )
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
                with tarfile.open(str(shard_path), "w") as tf:
                    for key in ("s0",):
                        for ext, data in [("png", png), ("cls", label.encode())]:
                            info = tarfile.TarInfo(name=f"{key}.{ext}")
                            buf = io.BytesIO(data)
                            info.size = len(data)
                            tf.addfile(info, buf)
        dm = _make_dm(
            root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        with pytest.raises(KeyError):
            dm.setup("fit")


class TestPriors:
    def test_priors_auto_computed_sums_to_one_after_setup(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("fit")
        assert dm.priors is not None
        assert abs(sum(dm.priors) - 1.0) < 1e-6
        assert all(p > 0 for p in dm.priors)

    def test_explicit_priors_preserved_after_setup(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        explicit = [0.3, 0.7]
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
            priors=explicit,
        )
        dm.setup("fit")
        assert dm.priors == explicit


class TestDataloaders:
    def _batch_from_loader(self, loader) -> dict:
        for batch in loader:
            return batch
        raise AssertionError("loader produced no batches")

    def test_train_dataloader_returns_webloader_with_correct_batch_keys(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        assert isinstance(loader, wds.WebLoader)
        batch = self._batch_from_loader(loader)
        assert set(batch.keys()) >= {"input", "target", "key"}

    def test_train_dataloader_batch_shapes_and_types(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = self._batch_from_loader(loader)
        assert batch["input"].ndim == 4
        assert batch["input"].dtype == torch.float32
        assert batch["target"].ndim == 1
        assert batch["target"].dtype == torch.int64
        assert isinstance(batch["key"], list)

    def test_train_dataloader_target_values_are_valid_class_indices(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = self._batch_from_loader(loader)
        targets = batch["target"]
        assert (targets >= 0).all() == True  # noqa: E712
        assert (targets < dm.num_classes).all() == True  # noqa: E712

    def test_val_dataloader_returns_webloader_with_correct_batch_keys(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("fit")
        loader = dm.val_dataloader()
        assert isinstance(loader, wds.WebLoader)
        batch = self._batch_from_loader(loader)
        assert set(batch.keys()) >= {"input", "target", "key"}

    def test_test_dataloader_returns_webloader_with_correct_batch_keys(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            train_loader_partial,
            eval_loader_partial,
        )
        dm.setup("test")
        loader = dm.test_dataloader()
        assert isinstance(loader, wds.WebLoader)
        batch = self._batch_from_loader(loader)
        assert set(batch.keys()) >= {"input", "target", "key"}

    def test_two_etl_labels_mapping_to_same_class_produce_same_index(
        self,
        shard_root,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        shared_map = {"NORMAL": "healthy", "ABNORMAL": "healthy"}
        dm = WebDatasetDataModule(
            shard_root=str(shard_root),
            label_map=shared_map,
            train_transform=train_transform,
            eval_transform=eval_transform,
            train_loader=train_loader_partial,
            eval_loader=eval_loader_partial,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = self._batch_from_loader(loader)
        targets = batch["target"]
        assert (targets == 0).all() == True  # noqa: E712
