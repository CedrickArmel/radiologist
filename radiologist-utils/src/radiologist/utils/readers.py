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
            with Image.open(path) as img:
                array = np.asarray(img.copy())
            yield array, {"path": str(path), "filename": path.name}


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
            with fs.open(file_path, "rb") as f:
                img = Image.open(f)
                img.load()
                array = np.asarray(img.copy())
            yield array, {"path": file_path, "filename": Path(file_path).name}


def ImageReader(
    source: str, storage_options: dict | None = None
) -> BaseImageReader:
    """Factory that returns a :class:`LocalImageReader` or :class:`RemoteImageReader`.

    Resolution is based on the URI scheme:
    - ``gs://``, ``gcs://``, ``s3://``, ``az://``, ``abfs://`` → remote
    - everything else → local
    """
    scheme = urlparse(source).scheme
    if scheme in _REMOTE_SCHEMES:
        return RemoteImageReader(source, storage_options=storage_options)
    return LocalImageReader(source)
