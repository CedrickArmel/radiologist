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

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import fsspec
import numpy as np
from PIL import Image

SUPPORTED_FORMATS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
_REMOTE_SCHEMES: frozenset[str] = frozenset({"gs", "gcs", "s3", "az", "abfs"})

ImageRecord = tuple[np.ndarray, dict[str, str]]


class BaseImageReader(ABC):
    """Abstract base for lazy, format-agnostic image readers."""

    def __init__(self, source: str) -> None:
        self.source = source

    @abstractmethod
    def iterate(self) -> Iterator[ImageRecord]:
        """Yield ``(array, metadata)`` tuples for each supported image."""
        ...


def read_image(source: str, storage_options: dict | None = None) -> ImageRecord:
    """Read a single PNG/JPEG image from a local path or remote URI.

    Args:
        source: local filesystem path or remote URI (e.g. ``gs://bucket/scan.png``).
        storage_options: extra kwargs forwarded to fsspec (e.g. GCS token).

    Returns:
        ``(array, metadata)`` where metadata contains ``"path"`` and ``"filename"``.
    """
    fs, path = fsspec.url_to_fs(source, **(storage_options or {}))
    if not fs.exists(path):
        raise FileNotFoundError(f"Image not found: {source!r}")
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format {suffix!r}")
    with fs.open(path, "rb") as f:
        img = Image.open(f)
        img.load()
        array = np.asarray(img.copy())
    return array, {"path": source, "filename": Path(path).name}


class LocalImageReader(BaseImageReader):
    """Stream PNG/JPEG images from a local directory tree."""

    def iterate(self) -> Iterator[ImageRecord]:
        root = Path(self.source)
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {self.source!r}")
        if not root.is_dir():
            raise NotADirectoryError(f"Expected a directory, got: {self.source!r}")
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_FORMATS:
                continue
            yield read_image(str(path))


class RemoteImageReader(BaseImageReader):
    """Stream PNG/JPEG images from a remote filesystem via fsspec (GCS, S3, …).

    Credentials are resolved from the environment (e.g. ``GOOGLE_APPLICATION_CREDENTIALS``).
    Pass ``storage_options`` to override or supply additional fsspec parameters.
    """

    def __init__(self, source: str, storage_options: dict | None = None) -> None:
        super().__init__(source)
        self._storage_options: dict = storage_options or {}

    def iterate(self) -> Iterator[ImageRecord]:
        fs, root = fsspec.url_to_fs(self.source, **self._storage_options)
        if not fs.exists(root):
            raise FileNotFoundError(f"Remote path not found: {self.source!r}")
        for file_path in sorted(fs.find(root)):
            if Path(file_path).suffix.lower() not in SUPPORTED_FORMATS:
                continue
            yield read_image(
                fs.unstrip_protocol(file_path), storage_options=self._storage_options
            )


def ImageReader(source: str, storage_options: dict | None = None) -> BaseImageReader:
    """Factory that returns a :class:`LocalImageReader` or :class:`RemoteImageReader`.

    Resolution is based on the URI scheme:
    - ``gs://``, ``gcs://``, ``s3://``, ``az://``, ``abfs://`` → remote
    - everything else → local
    """
    scheme = urlparse(source).scheme
    if scheme in _REMOTE_SCHEMES:
        return RemoteImageReader(source, storage_options=storage_options)
    return LocalImageReader(source)
