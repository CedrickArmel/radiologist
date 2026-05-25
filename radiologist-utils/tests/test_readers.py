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
    read_image,
)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


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
def mock_remote_fs():
    """Return an in-memory mock of an fsspec filesystem with one PNG file."""
    fs = MagicMock()
    remote_root = "bucket/radiology"
    image_path = f"{remote_root}/scan.png"

    fs.exists.return_value = True
    fs.find.return_value = [image_path]
    fs.unstrip_protocol.side_effect = lambda p: f"gs://{p}"

    return fs, remote_root


def test_base_reader_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseImageReader(source="/any")  # type: ignore[abstract]


def test_local_reader_returns_arrays_and_metadata(image_dir: Path) -> None:
    reader = LocalImageReader(source=str(image_dir))
    results = list(reader.iterate())

    assert len(results) == 2
    for array, meta in results:
        assert isinstance(array, np.ndarray)
        assert array.shape == (10, 10, 3)
        assert "path" in meta
        assert "filename" in meta


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


_DUMMY_RECORD = (
    np.zeros((10, 10, 3), dtype=np.uint8),
    {"path": "gs://bucket/radiology/scan.png", "filename": "scan.png"},
)


def test_remote_reader_returns_arrays_and_metadata(mock_remote_fs) -> None:
    fs, remote_root = mock_remote_fs
    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, remote_root)
    ):
        with patch("radiologist.utils.readers.read_image", return_value=_DUMMY_RECORD):
            results = list(RemoteImageReader(source="gs://bucket/radiology").iterate())

    assert len(results) == 1
    _, meta = results[0]
    assert meta["filename"] == "scan.png"


def test_remote_reader_missing_path_raises() -> None:
    fs = MagicMock()
    fs.exists.return_value = False
    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "bucket/bad")
    ):
        with pytest.raises(FileNotFoundError, match="gs://bucket/bad"):
            list(RemoteImageReader(source="gs://bucket/bad").iterate())


def test_remote_reader_forwards_storage_options() -> None:
    fs = MagicMock()
    fs.exists.return_value = False
    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "p")
    ) as mock_fn:
        try:
            list(
                RemoteImageReader(
                    source="gs://b/p", storage_options={"token": "anon"}
                ).iterate()
            )
        except FileNotFoundError:
            pass
        mock_fn.assert_called_once_with("gs://b/p", token="anon")


def test_remote_reader_skips_non_image_files() -> None:
    fs = MagicMock()
    fs.exists.return_value = True
    fs.find.return_value = ["bucket/path/doc.pdf", "bucket/path/img.png"]
    fs.unstrip_protocol.side_effect = lambda p: f"gs://{p}"

    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "bucket/path")
    ):
        with patch(
            "radiologist.utils.readers.read_image", return_value=_DUMMY_RECORD
        ) as mock_read:
            list(RemoteImageReader(source="gs://bucket/path").iterate())

    mock_read.assert_called_once_with("gs://bucket/path/img.png", storage_options={})


def test_remote_reader_calls_read_image_with_full_uri(mock_remote_fs) -> None:
    fs, remote_root = mock_remote_fs
    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, remote_root)
    ):
        with patch(
            "radiologist.utils.readers.read_image", return_value=_DUMMY_RECORD
        ) as mock_read:
            list(RemoteImageReader(source="gs://bucket/radiology").iterate())

    mock_read.assert_called_once_with(
        "gs://bucket/radiology/scan.png", storage_options={}
    )


@pytest.mark.parametrize("source", ["/local/path", "./relative", "no-scheme"])
def test_factory_returns_local_reader_for_local_paths(source: str) -> None:
    assert isinstance(ImageReader(source=source), LocalImageReader)


@pytest.mark.parametrize(
    "source", ["gs://bucket/path", "gcs://bucket/path", "s3://bucket/path"]
)
def test_factory_returns_remote_reader_for_remote_uris(source: str) -> None:
    assert isinstance(ImageReader(source=source), RemoteImageReader)


def test_factory_passes_storage_options_to_remote() -> None:
    opts = {"token": "anon"}
    reader = ImageReader(source="gs://b/p", storage_options=opts)
    assert reader._storage_options == opts  # type: ignore[union-attr]


def test_read_image_local_returns_array_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "scan.png"
    path.write_bytes(_png_bytes())

    array, meta = read_image(str(path))

    assert isinstance(array, np.ndarray)
    assert array.shape == (10, 10, 3)
    assert meta["filename"] == "scan.png"
    assert meta["path"] == str(path)


def test_read_image_local_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not_there.png"):
        read_image(str(tmp_path / "not_there.png"))


def test_read_image_unsupported_format_raises(tmp_path: Path) -> None:
    (tmp_path / "doc.pdf").write_bytes(b"%PDF")
    with pytest.raises(ValueError, match=".pdf"):
        read_image(str(tmp_path / "doc.pdf"))


def test_read_image_remote_returns_array_and_metadata() -> None:
    fs = MagicMock()
    fs.exists.return_value = True
    ctx = MagicMock()
    ctx.__enter__ = lambda s: io.BytesIO(_png_bytes())
    ctx.__exit__ = MagicMock(return_value=False)
    fs.open.return_value = ctx

    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs",
        return_value=(fs, "bucket/scan.png"),
    ):
        array, meta = read_image("gs://bucket/scan.png")

    assert isinstance(array, np.ndarray)
    assert meta["filename"] == "scan.png"
    assert meta["path"] == "gs://bucket/scan.png"


def test_read_image_remote_missing_raises() -> None:
    fs = MagicMock()
    fs.exists.return_value = False
    with patch(
        "radiologist.utils.readers.fsspec.url_to_fs", return_value=(fs, "bucket/x.png")
    ):
        with pytest.raises(FileNotFoundError, match="gs://bucket/x.png"):
            read_image("gs://bucket/x.png")
