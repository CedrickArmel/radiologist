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

import warnings
from pathlib import PurePath, PurePosixPath

import fsspec

warnings.filterwarnings("once")


def pathjoin(a: str, /, *paths: str) -> str:
    """Join path components, preserving the filesystem protocol."""
    fs, root = fsspec.url_to_fs(a)
    pure = PurePosixPath(root) if "local" not in fs.protocol else PurePath(root)
    return (
        fs.unstrip_protocol(str(pure.joinpath(*paths)))
        if "local" not in fs.protocol
        else str(pure.joinpath(*paths))
    )


def pathname(path: str) -> str:
    fs, root = fsspec.url_to_fs(path)
    return PurePosixPath(root).name


def pathparent(path: str) -> str:
    fs, root = fsspec.url_to_fs(path)
    root = PurePosixPath(root) if "local" not in fs.protocol else PurePath(root)
    parent = str(root.parent)
    return fs.unstrip_protocol(parent) if "local" not in fs.protocol else str(parent)


def pathstem(path) -> str:
    fs, root = fsspec.url_to_fs(path)
    return PurePosixPath(root).stem


__all__ = [
    "pathjoin",
    "pathname",
    "pathparent",
    "pathstem",
]
