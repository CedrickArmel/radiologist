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

"""Behavioral tests for the content/config/directory digest and per-stage
run-id computation seam (radiologist.etl.identity).
"""

from __future__ import annotations

from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import pytest

# --- content_digest ---------------------------------------------------------


def test_content_digest_is_deterministic_for_the_same_bytes(tmp_path: Path) -> None:
    from radiologist.etl import content_digest

    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    d1 = content_digest(str(p))
    d2 = content_digest(str(p))
    assert d1 == d2


def test_content_digest_changes_when_a_single_byte_changes(tmp_path: Path) -> None:
    from radiologist.etl import content_digest

    p1 = tmp_path / "a.bin"
    p1.write_bytes(b"hello world")
    p2 = tmp_path / "b.bin"
    p2.write_bytes(b"hello worle")

    assert content_digest(str(p1)) != content_digest(str(p2))


def test_content_digest_over_multiple_chunks_matches_single_small_object(
    tmp_path: Path,
) -> None:
    from radiologist.etl import content_digest

    payload = b"x" * 3000
    big = tmp_path / "big.bin"
    big.write_bytes(payload)
    small = tmp_path / "small.bin"
    small.write_bytes(payload)

    # force multiple read chunks on the "big" one
    digest_chunked = content_digest(str(big), chunk_size=256)
    digest_single = content_digest(str(small), chunk_size=1_048_576)
    assert digest_chunked == digest_single


def test_content_digest_raises_file_not_found_naming_the_missing_uri(
    tmp_path: Path,
) -> None:
    from radiologist.etl import content_digest

    missing = str(tmp_path / "nope.bin")
    with pytest.raises(FileNotFoundError, match="nope.bin"):
        content_digest(missing)


# --- config_digest -----------------------------------------------------------


def test_config_digest_unchanged_when_top_level_keys_reordered() -> None:
    from radiologist.etl.identity import config_digest

    d1 = config_digest({"a": 1, "b": 2})
    d2 = config_digest({"b": 2, "a": 1})
    assert d1 == d2


def test_config_digest_changes_when_a_value_changes() -> None:
    from radiologist.etl.identity import config_digest

    d1 = config_digest({"a": 1, "b": 2})
    d2 = config_digest({"a": 1, "b": 3})
    assert d1 != d2


def test_config_digest_unchanged_when_nested_keys_reordered() -> None:
    from radiologist.etl.identity import config_digest

    d1 = config_digest({"a": {"x": 1, "y": 2}, "b": 2})
    d2 = config_digest({"a": {"y": 2, "x": 1}, "b": 2})
    assert d1 == d2


# --- directory_digest ---------------------------------------------------------


def _write_jsonl(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_directory_digest_changes_when_a_manifest_is_added(tmp_path: Path) -> None:
    from radiologist.etl.identity import directory_digest

    d = tmp_path / "manifests"
    _write_jsonl(d / "a.jsonl", b'{"x": 1}\n')
    before = directory_digest(str(d))
    _write_jsonl(d / "b.jsonl", b'{"y": 2}\n')
    after = directory_digest(str(d))
    assert before != after


def test_directory_digest_changes_when_a_manifest_is_removed(tmp_path: Path) -> None:
    from radiologist.etl.identity import directory_digest

    d = tmp_path / "manifests"
    _write_jsonl(d / "a.jsonl", b'{"x": 1}\n')
    _write_jsonl(d / "b.jsonl", b'{"y": 2}\n')
    before = directory_digest(str(d))
    (d / "b.jsonl").unlink()
    after = directory_digest(str(d))
    assert before != after


def test_directory_digest_changes_when_a_manifest_size_changes(tmp_path: Path) -> None:
    from radiologist.etl.identity import directory_digest

    d = tmp_path / "manifests"
    _write_jsonl(d / "a.jsonl", b'{"x": 1}\n')
    before = directory_digest(str(d))
    _write_jsonl(d / "a.jsonl", b'{"x": 1, "longer": true}\n')
    after = directory_digest(str(d))
    assert before != after


def test_directory_digest_ignores_files_not_matching_suffix(tmp_path: Path) -> None:
    from radiologist.etl.identity import directory_digest

    d = tmp_path / "manifests"
    _write_jsonl(d / "a.jsonl", b'{"x": 1}\n')
    before = directory_digest(str(d), suffix=".jsonl")
    _write_jsonl(d / "notes.txt", b"irrelevant content, arbitrary length")
    after = directory_digest(str(d), suffix=".jsonl")
    assert before == after


def test_directory_digest_uses_a_single_detailed_listing_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """identity.py itself must issue exactly one listing call to the
    filesystem abstraction — never a listing followed by a per-entry stat
    call of its own. (A given fsspec backend's internal implementation of
    that one call is out of our control and out of scope here.)"""
    from radiologist.etl.identity import directory_digest

    d = tmp_path / "manifests"
    _write_jsonl(d / "a.jsonl", b'{"x": 1}\n')
    _write_jsonl(d / "b.jsonl", b'{"y": 2}\n')

    fs = fsspec.filesystem("file")
    calls = {"ls": 0}
    real_ls = type(fs).ls

    def counting_ls(self, *a, **kw):  # type: ignore[no-untyped-def]
        calls["ls"] += 1
        return real_ls(self, *a, **kw)

    monkeypatch.setattr(type(fs), "ls", counting_ls)

    directory_digest(str(d))

    assert calls["ls"] == 1


def test_directory_digest_raises_file_not_found_naming_the_missing_uri(
    tmp_path: Path,
) -> None:
    from radiologist.etl.identity import directory_digest

    missing = str(tmp_path / "no-such-dir")
    with pytest.raises(FileNotFoundError, match="no-such-dir"):
        directory_digest(missing)


# --- compute_extract_run_id ---------------------------------------------------


def _write_listing(path: Path, lines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_extract_run_id_is_the_same_across_calls_for_identical_inputs(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_extract_run_id

    listing = tmp_path / "files.txt"
    _write_listing(listing, ["/a.png", "/b.png"])
    config = {"quality": {"iqr_k": 1.5}}

    id1 = compute_extract_run_id(str(listing), config)
    id2 = compute_extract_run_id(str(listing), config)
    assert id1 == id2


def test_extract_run_id_changes_when_listing_contents_change(tmp_path: Path) -> None:
    from radiologist.etl import compute_extract_run_id

    listing = tmp_path / "files.txt"
    _write_listing(listing, ["/a.png", "/b.png"])
    config = {"quality": {"iqr_k": 1.5}}
    before = compute_extract_run_id(str(listing), config)

    _write_listing(listing, ["/a.png", "/b.png", "/c.png"])
    after = compute_extract_run_id(str(listing), config)
    assert before != after


def test_extract_run_id_changes_when_output_affecting_config_changes(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_extract_run_id

    listing = tmp_path / "files.txt"
    _write_listing(listing, ["/a.png", "/b.png"])
    before = compute_extract_run_id(str(listing), {"quality": {"iqr_k": 1.5}})
    after = compute_extract_run_id(str(listing), {"quality": {"iqr_k": 3.0}})
    assert before != after


def test_extract_run_id_unaffected_by_settings_outside_the_config_mapping(
    tmp_path: Path,
) -> None:
    """Worker count / batch size / runner family are never part of the config
    mapping this function hashes — callers exclude execution settings before
    calling. Calling with the same output-affecting config twice, as would
    happen when only execution settings differ between two invocations,
    always yields the same id."""
    from radiologist.etl import compute_extract_run_id

    listing = tmp_path / "files.txt"
    _write_listing(listing, ["/a.png", "/b.png"])
    config = {"quality": {"iqr_k": 1.5}}

    id_with_4_workers = compute_extract_run_id(str(listing), config)
    id_with_16_workers = compute_extract_run_id(str(listing), config)
    assert id_with_4_workers == id_with_16_workers


# --- compute_assign_run_id -----------------------------------------------------


def test_assign_run_id_is_the_same_across_calls_for_identical_inputs(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_assign_run_id

    d = tmp_path / "manifests"
    _write_jsonl(d / "extract-1.jsonl", b'{"x": 1}\n')
    ratios = {"ratios": [["train", 0.8], ["val", 0.1], ["test", 0.1]]}

    id1 = compute_assign_run_id(str(d), ratios)
    id2 = compute_assign_run_id(str(d), ratios)
    assert id1 == id2


def test_assign_run_id_changes_when_a_manifest_is_added(tmp_path: Path) -> None:
    from radiologist.etl import compute_assign_run_id

    d = tmp_path / "manifests"
    _write_jsonl(d / "extract-1.jsonl", b'{"x": 1}\n')
    ratios = {"ratios": [["train", 0.8], ["val", 0.1], ["test", 0.1]]}
    before = compute_assign_run_id(str(d), ratios)

    _write_jsonl(d / "extract-2.jsonl", b'{"y": 2}\n')
    after = compute_assign_run_id(str(d), ratios)
    assert before != after


def test_assign_run_id_changes_when_ratio_order_changes(tmp_path: Path) -> None:
    from radiologist.etl import compute_assign_run_id

    d = tmp_path / "manifests"
    _write_jsonl(d / "extract-1.jsonl", b'{"x": 1}\n')

    ordered = {"ratios": [["train", 0.8], ["val", 0.1], ["test", 0.1]]}
    reordered = {"ratios": [["val", 0.1], ["train", 0.8], ["test", 0.1]]}

    id_ordered = compute_assign_run_id(str(d), ordered)
    id_reordered = compute_assign_run_id(str(d), reordered)
    assert id_ordered != id_reordered


# --- compute_build_run_id -------------------------------------------------------


def test_build_run_id_is_the_same_across_calls_for_identical_inputs(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_build_run_id

    manifest = tmp_path / "split-manifest.jsonl"
    _write_jsonl(manifest, b'{"path": "/a.png", "split": "train"}\n')
    config = {"shard_size": 500}

    id1 = compute_build_run_id(str(manifest), config)
    id2 = compute_build_run_id(str(manifest), config)
    assert id1 == id2


def test_build_run_id_changes_when_manifest_contents_change(tmp_path: Path) -> None:
    from radiologist.etl import compute_build_run_id

    manifest = tmp_path / "split-manifest.jsonl"
    _write_jsonl(manifest, b'{"path": "/a.png", "split": "train"}\n')
    config = {"shard_size": 500}
    before = compute_build_run_id(str(manifest), config)

    _write_jsonl(manifest, b'{"path": "/a.png", "split": "val"}\n')
    after = compute_build_run_id(str(manifest), config)
    assert before != after


def test_build_run_id_changes_when_shard_size_changes(tmp_path: Path) -> None:
    from radiologist.etl import compute_build_run_id

    manifest = tmp_path / "split-manifest.jsonl"
    _write_jsonl(manifest, b'{"path": "/a.png", "split": "train"}\n')
    before = compute_build_run_id(str(manifest), {"shard_size": 500})
    after = compute_build_run_id(str(manifest), {"shard_size": 1000})
    assert before != after


# --- cross-stage invariants -----------------------------------------------------


def test_run_ids_are_16_characters_long(tmp_path: Path) -> None:
    from radiologist.etl import compute_build_run_id

    manifest = tmp_path / "split-manifest.jsonl"
    _write_jsonl(manifest, b'{"path": "/a.png", "split": "train"}\n')
    run_id = compute_build_run_id(str(manifest), {"shard_size": 500})
    assert len(run_id) == 16


def test_run_ids_differ_across_stages_for_the_same_underlying_inputs(
    tmp_path: Path,
) -> None:
    """A file whose bytes are identical to a manifest folder listing digest
    input should still produce different ids per stage, because the stage
    name is mixed into the hashed payload."""
    from radiologist.etl import (
        compute_assign_run_id,
        compute_build_run_id,
        compute_extract_run_id,
    )

    shared_path = tmp_path / "shared.jsonl"
    _write_jsonl(shared_path, b'{"a": 1}\n')

    listing = tmp_path / "listing.txt"
    _write_listing(listing, ["/a.png"])

    manifests_dir = tmp_path / "manifests"
    _write_jsonl(manifests_dir / "shared.jsonl", b'{"a": 1}\n')

    config: dict = {}
    extract_id = compute_extract_run_id(str(shared_path), config)
    build_id = compute_build_run_id(str(shared_path), config)
    assign_id = compute_assign_run_id(str(manifests_dir), config)

    assert len({extract_id, build_id, assign_id}) == 3


# --- run-id error handling -------------------------------------------------------


def test_compute_extract_run_id_raises_file_not_found_naming_missing_uri(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_extract_run_id

    missing = str(tmp_path / "no-listing.txt")
    with pytest.raises(FileNotFoundError, match="no-listing.txt"):
        compute_extract_run_id(missing, {})


def test_compute_assign_run_id_raises_file_not_found_naming_missing_uri(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_assign_run_id

    missing = str(tmp_path / "no-such-manifests-dir")
    with pytest.raises(FileNotFoundError, match="no-such-manifests-dir"):
        compute_assign_run_id(missing, {})


def test_compute_build_run_id_raises_file_not_found_naming_missing_uri(
    tmp_path: Path,
) -> None:
    from radiologist.etl import compute_build_run_id

    missing = str(tmp_path / "no-manifest.jsonl")
    with pytest.raises(FileNotFoundError, match="no-manifest.jsonl"):
        compute_build_run_id(missing, {})


# --- storage_options threading ------------------------------------------------


def test_content_digest_forwards_storage_options_to_every_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem that requires a specific storage option resolves
    successfully only when that option reaches every fsspec call the digest
    helper performs."""
    from radiologist.etl import content_digest

    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")

    seen_options: list = []
    real_url_to_fs = fsspec.url_to_fs

    def spying_url_to_fs(uri, **kwargs):  # type: ignore[no-untyped-def]
        seen_options.append(kwargs)
        return real_url_to_fs(uri, **kwargs)

    monkeypatch.setattr(fsspec, "url_to_fs", spying_url_to_fs)

    content_digest(str(p), storage_options={"anon": True})

    assert all(opts.get("anon") is True for opts in seen_options)


def test_directory_digest_forwards_storage_options_to_every_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from radiologist.etl.identity import directory_digest

    d = tmp_path / "manifests"
    _write_jsonl(d / "a.jsonl", b'{"x": 1}\n')

    seen_options: list = []
    real_url_to_fs = fsspec.url_to_fs

    def spying_url_to_fs(uri, **kwargs):  # type: ignore[no-untyped-def]
        seen_options.append(kwargs)
        return real_url_to_fs(uri, **kwargs)

    monkeypatch.setattr(fsspec, "url_to_fs", spying_url_to_fs)

    directory_digest(str(d), storage_options={"anon": True})

    assert all(opts.get("anon") is True for opts in seen_options)


# --- records_reader signature contract -----------------------------------------


def test_records_reader_reads_jsonl_without_storage_options(tmp_path: Path) -> None:
    from radiologist.etl import JsonlWriter, ManifestRecord, records_reader

    rec = ManifestRecord(
        manifest_id="run-abc-0000001",
        path="/data/NORMAL/scan.png",
        filename="scan.png",
        label="NORMAL",
        split="train",
        stats={"haralick_contrast": 0.5},
    )
    dest = str(tmp_path / "manifest.jsonl")
    JsonlWriter().write([rec], dest)

    records = records_reader(dest)
    assert len(records) == 1
    assert records[0].manifest_id == "run-abc-0000001"


def test_records_reader_accepts_storage_options_positionally(tmp_path: Path) -> None:
    from radiologist.etl import JsonlWriter, ManifestRecord, records_reader

    rec = ManifestRecord(
        manifest_id="run-abc-0000001",
        path="/data/NORMAL/scan.png",
        filename="scan.png",
        label="NORMAL",
        split="train",
        stats={"haralick_contrast": 0.5},
    )
    dest = str(tmp_path / "manifest.jsonl")
    JsonlWriter().write([rec], dest)

    records_no_opts = records_reader(dest)
    records_with_opts = records_reader(dest, {})
    assert records_with_opts == records_no_opts
