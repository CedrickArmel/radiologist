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

from __future__ import annotations

import pytest

from radiologist.utils.filesystem import pathjoin, pathname, pathparent, pathstem

LOCAL = "local"
REMOTE = "remote"

REMOTE_SCHEMES = ["gs"]


def remote(scheme: str, path: str) -> str:
    return f"{scheme}://bucket/{path}"


class TestPathjoin:
    def test_local(self) -> None:
        assert pathjoin("/foo", "bar", "baz.txt") == "/foo/bar/baz.txt"

    def test_local_single_segment(self) -> None:
        assert pathjoin("/foo", "bar") == "/foo/bar"

    def test_local_base_only(self) -> None:
        assert pathjoin("/foo/bar") == "/foo/bar"

    @pytest.mark.parametrize("scheme", REMOTE_SCHEMES)
    def test_remote(self, scheme: str) -> None:
        assert (
            pathjoin(f"{scheme}://bucket", "dir", "file.txt")
            == f"{scheme}://bucket/dir/file.txt"
        )

    @pytest.mark.parametrize("scheme", REMOTE_SCHEMES)
    def test_remote_nested(self, scheme: str) -> None:
        assert (
            pathjoin(f"{scheme}://bucket", "a", "b", "c.csv")
            == f"{scheme}://bucket/a/b/c.csv"
        )


class TestPathname:
    def test_local(self) -> None:
        assert pathname("/foo/bar/baz.txt") == "baz.txt"

    def test_local_multiple_extensions(self) -> None:
        assert pathname("/foo/bar/archive.tar.gz") == "archive.tar.gz"

    def test_local_no_extension(self) -> None:
        assert pathname("/foo/bar/mydir") == "mydir"

    @pytest.mark.parametrize("scheme", REMOTE_SCHEMES)
    def test_remote(self, scheme: str) -> None:
        assert pathname(remote(scheme, "dir/file.txt")) == "file.txt"


class TestPathparent:
    def test_local(self) -> None:
        assert pathparent("/foo/bar/baz.txt") == "/foo/bar"

    def test_local_root(self) -> None:
        assert pathparent("/baz.txt") == "/"

    @pytest.mark.parametrize("scheme", REMOTE_SCHEMES)
    def test_remote(self, scheme: str) -> None:
        assert pathparent(remote(scheme, "dir/file.txt")) == f"{scheme}://bucket/dir"

    @pytest.mark.parametrize("scheme", REMOTE_SCHEMES)
    def test_remote_nested(self, scheme: str) -> None:
        assert pathparent(remote(scheme, "a/b/c.csv")) == f"{scheme}://bucket/a/b"


class TestPathstem:
    def test_local(self) -> None:
        assert pathstem("/foo/bar/baz.txt") == "baz"

    def test_local_last_extension_only(self) -> None:
        assert pathstem("/foo/bar/archive.tar.gz") == "archive.tar"

    def test_local_no_extension(self) -> None:
        assert pathstem("/foo/bar/mydir") == "mydir"

    @pytest.mark.parametrize("scheme", REMOTE_SCHEMES)
    def test_remote(self, scheme: str) -> None:
        assert pathstem(remote(scheme, "dir/file.txt")) == "file"
