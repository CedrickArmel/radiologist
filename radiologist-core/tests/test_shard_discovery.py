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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from radiologist.core.data.shards import _discover_shards  # noqa: E402


class TestLocalDiscoveryPreservesExistingBehavior:
    def test_local_glob_layout_returns_per_split_per_label_mapping(
        self, shard_root: Path
    ) -> None:
        result = _discover_shards(str(shard_root), ["train", "val"])
        assert set(result.keys()) == {"train", "val"}
        assert set(result["train"].keys()) == {"NORMAL", "ABNORMAL"}
        assert set(result["val"].keys()) == {"NORMAL", "ABNORMAL"}
        assert len(result["train"]["NORMAL"]) >= 1
        assert len(result["train"]["ABNORMAL"]) >= 1

    def test_missing_split_raises_file_not_found_naming_the_split(
        self, tmp_path: Path
    ) -> None:
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        with pytest.raises(FileNotFoundError, match="train"):
            _discover_shards(str(empty_root), ["train"])


class TestExpandSpecBraceRange:
    def test_brace_range_expands_to_exact_count_without_filesystem_access(
        self,
    ) -> None:
        from radiologist.core.data.shards import _expand_spec

        spec = "file:///tmp/data/{0000..0002}.tar"
        result = _expand_spec(spec)
        assert result == [
            "file:///tmp/data/0000.tar",
            "file:///tmp/data/0001.tar",
            "file:///tmp/data/0002.tar",
        ]

    def test_brace_list_expands_to_named_variants(self) -> None:
        from radiologist.core.data.shards import _expand_spec

        spec = "s3://bucket/{train,val}/shard.tar"
        result = _expand_spec(spec)
        assert result == [
            "s3://bucket/train/shard.tar",
            "s3://bucket/val/shard.tar",
        ]


class TestExpandSpecWildcard:
    def test_wildcard_resolves_existing_local_paths(self, tmp_path: Path) -> None:
        from radiologist.core.data.shards import _expand_spec

        for name in ("a.tar", "b.tar"):
            (tmp_path / name).write_bytes(b"")
        result = _expand_spec(f"{tmp_path}/*.tar")
        assert sorted(result) == sorted(
            [str(tmp_path / "a.tar"), str(tmp_path / "b.tar")]
        )

    def test_literal_path_returned_as_is(self) -> None:
        from radiologist.core.data.shards import _expand_spec

        spec = "http://example.com/data/shard.tar"
        result = _expand_spec(spec)
        assert result == ["http://example.com/data/shard.tar"]


class TestDiscoverShardsWithBraceSpec:
    def test_brace_power_mode_discovers_matching_shards(self, shard_root: Path) -> None:
        spec = f"{shard_root}/{{train,val}}/*/*.tar"
        result = _discover_shards(spec, ["train", "val"])
        assert set(result.keys()) == {"train", "val"}
        assert set(result["train"].keys()) == {"NORMAL", "ABNORMAL"}

    def test_wildcard_power_mode_same_paths_as_plain_mode(
        self, shard_root: Path
    ) -> None:
        plain = _discover_shards(str(shard_root), ["train"])
        wildcard = _discover_shards(f"{shard_root}/train/*/*.tar", ["train"])
        assert plain["train"] == wildcard["train"]


class TestLabelInference:
    def test_label_inferred_correctly_for_posix_path(self, shard_root: Path) -> None:
        result = _discover_shards(str(shard_root), ["train"])
        assert "NORMAL" in result["train"]
        assert "ABNORMAL" in result["train"]

    def test_label_inferred_correctly_for_windows_style_path(self) -> None:
        from radiologist.core.data.shards import _label_from_path

        path = "C:\\data\\train\\NORMAL\\shard.tar"
        assert _label_from_path(path) == "NORMAL"

    def test_label_inferred_correctly_for_cloud_key_path(self) -> None:
        from radiologist.core.data.shards import _label_from_path

        path = "s3://bucket/split/label/shard.tar"
        assert _label_from_path(path) == "label"

    def test_label_inferred_correctly_for_posix_path_direct(self) -> None:
        from radiologist.core.data.shards import _label_from_path

        path = "/data/train/ABNORMAL/shard.tar"
        assert _label_from_path(path) == "ABNORMAL"
