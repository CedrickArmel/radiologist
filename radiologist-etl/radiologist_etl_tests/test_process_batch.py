# MIT License
#
# Copyright (c) 2026 @CedrickArmel
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

"""Behavioral tests for processors.process_batch (#183) — the picklable
per-batch worker the extract stage's mapper dispatches."""

from __future__ import annotations

import pytest


def test_readable_images_in_a_batch_each_produce_a_record(image_dir):
    from radiologist.etl.models import BatchOutcome
    from radiologist.etl.processors import process_batch
    from radiologist.etl.stats import make_haralick

    paths = [str(p) for p in sorted(image_dir.rglob("*.png"))]

    outcome = process_batch(
        paths,
        images_root=str(image_dir),
        masks_root=None,
        manifest_id="run-0000000000000001",
        extractors=[make_haralick(features=["mean"])],
    )

    assert isinstance(outcome, BatchOutcome)
    assert len(outcome.records) == len(paths)
    assert outcome.failures == []
    assert {r.path for r in outcome.records} == set(paths)


def test_unreadable_image_is_carried_as_a_failure_not_raised(image_dir, tmp_path):
    from radiologist.etl.processors import process_batch
    from radiologist.etl.stats import make_haralick

    good_paths = [str(p) for p in sorted(image_dir.rglob("*.png"))]
    bad_path = str(tmp_path / "missing.png")
    paths = good_paths + [bad_path]

    outcome = process_batch(
        paths,
        images_root=str(image_dir),
        masks_root=None,
        manifest_id="run-0000000000000001",
        extractors=[make_haralick(features=["mean"])],
    )

    assert len(outcome.records) == len(good_paths)
    assert len(outcome.failures) == 1
    failed_path, message = outcome.failures[0]
    assert failed_path == bad_path
    assert message


def test_records_carry_label_from_containing_folder(image_dir):
    from radiologist.etl.processors import process_batch
    from radiologist.etl.stats import make_haralick

    paths = [
        str(p) for p in sorted(image_dir.rglob("*.png")) if p.parent.name == "NORMAL"
    ]

    outcome = process_batch(
        paths,
        images_root=str(image_dir),
        masks_root=None,
        manifest_id="run-0000000000000001",
        extractors=[make_haralick(features=["mean"])],
    )

    assert all(r.label == "NORMAL" for r in outcome.records)
    assert all(r.split == "" for r in outcome.records)


def test_records_carry_lung_asymmetry_stats_when_mask_available(image_dir, mask_dir):
    from radiologist.etl.processors import process_batch
    from radiologist.etl.stats import lung_asymmetry, make_haralick

    paths = [str(p) for p in sorted(image_dir.rglob("*.png"))]

    outcome = process_batch(
        paths,
        images_root=str(image_dir),
        masks_root=str(mask_dir),
        manifest_id="run-0000000000000001",
        extractors=[make_haralick(features=["mean"]), lung_asymmetry],
    )

    assert all("asymmetry_ratio" in r.stats for r in outcome.records)
    assert all(r.lung_out_of_frame is not None for r in outcome.records)


def test_records_have_no_lung_out_of_frame_flag_without_masks(image_dir):
    from radiologist.etl.processors import process_batch
    from radiologist.etl.stats import make_haralick

    paths = [str(p) for p in sorted(image_dir.rglob("*.png"))]

    outcome = process_batch(
        paths,
        images_root=str(image_dir),
        masks_root=None,
        manifest_id="run-0000000000000001",
        extractors=[make_haralick(features=["mean"])],
    )

    assert all(r.lung_out_of_frame is None for r in outcome.records)


# --- masks_root/images_root invariant ----------------------------------------


def test_a_mask_root_without_an_images_root_is_rejected_naming_both_settings(
    image_dir, mask_dir
):
    from radiologist.etl import process_batch
    from radiologist.etl.stats import make_haralick

    paths = [str(p) for p in sorted(image_dir.rglob("*.png"))]

    with pytest.raises(ValueError) as excinfo:
        process_batch(
            paths,
            images_root=None,
            masks_root=str(mask_dir),
            manifest_id="run-0000000000000001",
            extractors=[make_haralick(features=["mean"])],
        )

    message = str(excinfo.value)
    assert "masks_root" in message
    assert "images_root" in message


def test_the_invalid_root_pair_is_rejected_before_any_image_is_read(tmp_path):
    from radiologist.etl import process_batch
    from radiologist.etl.stats import make_haralick

    absent = [str(tmp_path / "nowhere" / f"img{i}.png") for i in range(3)]

    for paths in ([], absent):
        with pytest.raises(ValueError, match="images_root"):
            process_batch(
                paths,
                images_root=None,
                masks_root=str(tmp_path / "masks"),
                manifest_id="run-0000000000000001",
                extractors=[make_haralick(features=["mean"])],
            )


def test_dispatching_beam_batches_without_an_images_root_surfaces_the_error(
    image_dir, mask_dir, tmp_path
):
    from radiologist.etl.beam_executor import BeamExecutor
    from radiologist.etl.stats import make_haralick

    paths = [str(p) for p in sorted(image_dir.rglob("*.png"))]
    executor = BeamExecutor(
        pipeline_options={"runner": "DirectRunner"},
        parts_dir=str(tmp_path / "parts"),
    )

    with pytest.raises(Exception) as excinfo:
        executor.run_batches(
            [paths],
            images_root=None,
            masks_root=str(mask_dir),
            manifest_id="run-0000000000000001",
            extractors=[make_haralick(features=["mean"])],
        )

    assert "images_root" in str(excinfo.value)
