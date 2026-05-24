from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from radiologist.utils.readers import (
    BaseImageReader,
    ImageReader,
    LocalImageReader,
    RemoteImageReader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.png").write_bytes(_png_bytes())
    (tmp_path / "b.jpg").write_bytes(_png_bytes())
    (tmp_path / "readme.txt").write_text("not an image")
    return tmp_path


@pytest.fixture()
def empty_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def mock_remote_fs(tmp_path: Path):
    """Return a fake fsspec-like filesystem backed by a real temp dir."""
    fs = MagicMock()
    remote_root = "bucket/radiology"
    image_path = f"{remote_root}/scan.png"

    fs.exists.return_value = True
    fs.find.return_value = [image_path]
    fs.open.return_value.__enter__ = lambda s: io.BytesIO(_png_bytes())
    fs.open.return_value.__exit__ = MagicMock(return_value=False)

    return fs, remote_root


# ---------------------------------------------------------------------------
# BaseImageReader contract
# ---------------------------------------------------------------------------


def test_base_reader_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseImageReader(source="/any")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# LocalImageReader
# ---------------------------------------------------------------------------


def test_local_reader_returns_arrays_and_metadata(image_dir: Path) -> None:
    reader = LocalImageReader(source=str(image_dir))
    results = list(reader.iterate())

    assert len(results) == 2
    for array, meta in results:
        assert isinstance(array, np.ndarray)
        assert array.shape == (10, 10, 3)
        assert "path" in meta
        assert "filename" in meta


def test_local_reader_skips_non_image_files(image_dir: Path) -> None:
    reader = LocalImageReader(source=str(image_dir))
    filenames = [meta["filename"] for _, meta in reader.iterate()]
    assert "readme.txt" not in filenames


def test_local_reader_empty_dir_yields_nothing(empty_dir: Path) -> None:
    reader = LocalImageReader(source=str(empty_dir))
    assert list(reader.iterate()) == []


def test_local_reader_missing_path_raises() -> None:
    reader = LocalImageReader(source="/does/not/exist")
    with pytest.raises(FileNotFoundError, match="/does/not/exist"):
        list(reader.iterate())


def test_local_reader_file_not_dir_raises(tmp_path: Path) -> None:
    f = tmp_path / "file.png"
    f.write_bytes(_png_bytes())
    reader = LocalImageReader(source=str(f))
    with pytest.raises(NotADirectoryError):
        list(reader.iterate())


# ---------------------------------------------------------------------------
# RemoteImageReader
# ---------------------------------------------------------------------------


def test_remote_reader_returns_arrays_and_metadata(mock_remote_fs) -> None:
    fs, remote_root = mock_remote_fs
    with patch("radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, remote_root)):
        reader = RemoteImageReader(source="gs://bucket/radiology")
        results = list(reader.iterate())

    assert len(results) == 1
    array, meta = results[0]
    assert isinstance(array, np.ndarray)
    assert meta["filename"] == "scan.png"
    assert meta["path"] == f"{remote_root}/scan.png"


def test_remote_reader_missing_path_raises() -> None:
    fs = MagicMock()
    fs.exists.return_value = False
    with patch("radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "bucket/bad")):
        reader = RemoteImageReader(source="gs://bucket/bad")
        with pytest.raises(FileNotFoundError, match="gs://bucket/bad"):
            list(reader.iterate())


def test_remote_reader_forwards_storage_options() -> None:
    fs = MagicMock()
    fs.exists.return_value = False
    with patch("radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "p")) as mock_fn:
        reader = RemoteImageReader(source="gs://b/p", storage_options={"token": "anon"})
        try:
            list(reader.iterate())
        except FileNotFoundError:
            pass
        mock_fn.assert_called_once_with("gs://b/p", token="anon")


def test_remote_reader_skips_non_image_files() -> None:
    fs = MagicMock()
    fs.exists.return_value = True
    fs.find.return_value = ["bucket/path/doc.pdf", "bucket/path/img.png"]

    def fake_open(path, mode="rb"):
        ctx = MagicMock()
        ctx.__enter__ = lambda s: io.BytesIO(_png_bytes())
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    fs.open.side_effect = fake_open

    with patch("radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "bucket/path")):
        reader = RemoteImageReader(source="gs://bucket/path")
        results = list(reader.iterate())

    assert len(results) == 1
    assert results[0][1]["filename"] == "img.png"


# ---------------------------------------------------------------------------
# ImageReader factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["/local/path", "./relative", "no-scheme"])
def test_factory_returns_local_reader_for_local_paths(source: str) -> None:
    assert isinstance(ImageReader(source=source), LocalImageReader)


@pytest.mark.parametrize("source", ["gs://bucket/path", "gcs://bucket/path", "s3://bucket/path"])
def test_factory_returns_remote_reader_for_remote_uris(source: str) -> None:
    assert isinstance(ImageReader(source=source), RemoteImageReader)


def test_factory_passes_storage_options_to_remote() -> None:
    opts = {"token": "anon"}
    reader = ImageReader(source="gs://b/p", storage_options=opts)
    assert isinstance(reader, RemoteImageReader)
    assert reader._storage_options == opts
