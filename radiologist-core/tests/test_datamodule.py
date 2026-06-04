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

from unittest.mock import MagicMock, patch

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
    batches_per_epoch: int = 10,
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
        batches_per_epoch=batches_per_epoch,
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
            batches_per_epoch=10,
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

    def test_priors_equal_per_class_sample_count_fractions(
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
        # shard_root has 2 ABNORMAL and 2 NORMAL train samples;
        # classes = ["abnormal", "normal"] -> each prior = 0.5
        assert dm.priors is not None
        assert len(dm.priors) == len(classes)
        for p in dm.priors:
            assert abs(p - 0.5) < 1e-6

    def test_shard_scan_runs_only_on_global_zero_rank_and_broadcast_delivers_priors(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        train_loader_partial,
        eval_loader_partial,
    ) -> None:
        expected_priors = [0.5, 0.5]

        fake_strategy = MagicMock()
        fake_strategy.broadcast.return_value = expected_priors

        fake_trainer_rank0 = MagicMock()
        fake_trainer_rank0.is_global_zero = True
        fake_trainer_rank0.strategy = fake_strategy

        fake_trainer_rank1 = MagicMock()
        fake_trainer_rank1.is_global_zero = False
        fake_trainer_rank1.strategy = fake_strategy

        with patch("radiologist.core.data.datamodule._count_samples") as mock_count:
            mock_count.return_value = 2

            dm_rank0 = _make_dm(
                shard_root,
                label_map,
                classes,
                train_transform,
                eval_transform,
                train_loader_partial,
                eval_loader_partial,
            )
            dm_rank0.trainer = fake_trainer_rank0
            dm_rank0.setup("fit")

            rank0_scan_calls = mock_count.call_count
            assert rank0_scan_calls > 0

            mock_count.reset_mock()

            dm_rank1 = _make_dm(
                shard_root,
                label_map,
                classes,
                train_transform,
                eval_transform,
                train_loader_partial,
                eval_loader_partial,
            )
            dm_rank1.trainer = fake_trainer_rank1
            dm_rank1.setup("fit")

            assert mock_count.call_count == 0
            assert dm_rank1.priors == expected_priors
            # rank-1 passes its placeholder (None) to broadcast and receives the value
            fake_strategy.broadcast.assert_called_with(None, src=0)

    def test_no_shard_scan_when_explicit_priors_provided(
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
        with patch("radiologist.core.data.datamodule._count_samples") as mock_count:
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
            assert mock_count.call_count == 0
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
            batches_per_epoch=10,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = self._batch_from_loader(loader)
        targets = batch["target"]
        assert (targets == 0).all() == True  # noqa: E712


class TestWebDatasetConstructorSplitArgs:
    """WebDataset must be constructed with nodesplitter=None, workersplitter=None.

    The constructor's built-in splitters (single_node_only + split_by_worker) must
    be disabled so that the explicit .compose(split_by_node, split_by_worker) is the
    single source of splitting — preventing double-splits and multi-node crashes.
    """

    def _collect_webdataset_kwargs(self, dm: WebDatasetDataModule, stage: str) -> list:
        """Call the appropriate dataloader and return captured WebDataset call kwargs."""
        captured = []
        original_init = wds.WebDataset.__init__

        def capturing_init(self_inner, *args, **kwargs):
            captured.append(kwargs)
            return original_init(self_inner, *args, **kwargs)

        with patch.object(wds.WebDataset, "__init__", capturing_init):
            if stage == "train":
                dm.train_dataloader()
            elif stage == "val":
                dm.val_dataloader()
            elif stage == "test":
                dm.test_dataloader()

        return captured

    def test_build_pipeline_passes_nodesplitter_none_to_webdataset(
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
        captured = self._collect_webdataset_kwargs(dm, "val")
        assert len(captured) >= 1
        for kwargs in captured:
            assert kwargs.get("nodesplitter") is None

    def test_build_pipeline_passes_workersplitter_none_to_webdataset(
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
        captured = self._collect_webdataset_kwargs(dm, "val")
        assert len(captured) >= 1
        for kwargs in captured:
            assert kwargs.get("workersplitter") is None

    def test_build_class_pipelines_passes_nodesplitter_none_to_webdataset(
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
        captured = self._collect_webdataset_kwargs(dm, "train")
        assert len(captured) >= 1
        for kwargs in captured:
            assert kwargs.get("nodesplitter") is None

    def test_build_class_pipelines_passes_workersplitter_none_to_webdataset(
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
        captured = self._collect_webdataset_kwargs(dm, "train")
        assert len(captured) >= 1
        for kwargs in captured:
            assert kwargs.get("workersplitter") is None


class TestBatchesPerEpoch:
    """WebDatasetDataModule must bound the training epoch to batches_per_epoch batches."""

    def test_train_epoch_terminates_after_exactly_batches_per_epoch(
        self,
        shard_root,
        label_map,
        classes,
        train_transform,
        eval_transform,
        eval_loader_partial,
    ) -> None:
        from functools import partial

        batches_per_epoch = 3
        bounded_loader = partial(wds.WebLoader, batch_size=1, num_workers=0)
        dm = _make_dm(
            shard_root,
            label_map,
            classes,
            train_transform,
            eval_transform,
            bounded_loader,
            eval_loader_partial,
            batches_per_epoch=batches_per_epoch,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        count = sum(1 for _ in loader)
        assert count == batches_per_epoch
