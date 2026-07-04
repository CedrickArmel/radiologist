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

"""Protocol-aware path helpers built on fsspec.

These functions manipulate local and remote (fsspec) paths uniformly,
preserving the URI scheme (e.g. ``gs://``, ``s3://``) for remote paths.
"""

import warnings
from pathlib import PurePath, PurePosixPath

import fsspec

warnings.filterwarnings("once")


def pathjoin(a: str, /, *paths: str) -> str:
    """Join path components, preserving the filesystem protocol.

    Args:
        a: Base local path or remote URI.
        *paths: Additional path components to append.

    Returns:
        The joined path, re-prefixed with the original protocol when ``a``
        is a remote URI.
    """
    fs, root = fsspec.url_to_fs(a)
    pure = PurePosixPath(root) if "local" not in fs.protocol else PurePath(root)
    return (
        fs.unstrip_protocol(str(pure.joinpath(*paths)))
        if "local" not in fs.protocol
        else str(pure.joinpath(*paths))
    )


def pathname(path: str) -> str:
    """Return the final path component (file or directory name).

    Args:
        path: Local path or remote URI.

    Returns:
        The last component of ``path``, without any protocol prefix.
    """
    fs, root = fsspec.url_to_fs(path)
    return PurePosixPath(root).name


def pathparent(path: str) -> str:
    """Return the parent directory of ``path``, preserving the protocol.

    Args:
        path: Local path or remote URI.

    Returns:
        The parent path, re-prefixed with the original protocol when
        ``path`` is a remote URI.
    """
    fs, root = fsspec.url_to_fs(path)
    root = PurePosixPath(root) if "local" not in fs.protocol else PurePath(root)
    parent = str(root.parent)
    return fs.unstrip_protocol(parent) if "local" not in fs.protocol else str(parent)


def pathstem(path) -> str:
    """Return the final path component without its suffix.

    Args:
        path: Local path or remote URI.

    Returns:
        The last path component with its file extension removed.
    """
    fs, root = fsspec.url_to_fs(path)
    return PurePosixPath(root).stem


__all__ = [
    "pathjoin",
    "pathname",
    "pathparent",
    "pathstem",
]
