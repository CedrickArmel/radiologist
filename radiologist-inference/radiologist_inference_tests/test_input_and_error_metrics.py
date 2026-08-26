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

"""Behavioural tests for the error taxonomy metric (4) and input-image
metrics (5, 6a, 6b).

Covers ``inference_errors_total``, ``inference_input_image_size_bytes``,
``inference_input_image_width_pixels`` and ``inference_input_image_height_pixels``,
driven exclusively through real HTTP traffic against a real ``TestClient``.
Assertions compare scrape deltas (scrape -> act -> scrape -> compare), never
absolute values.
"""

from typing import Any

import pytest
from _helpers import _hist, _make_png_bytes, _sample, build_det_onnx, build_mcd_onnx
from fastapi.testclient import TestClient

from radiologist.inference import Classifier, Explainer, MCDropoutPredictor, create_app


def _errors(client: TestClient, route: str, error_type: str) -> float:
    return _sample(
        client.get("/metrics").text,
        "inference_errors_total",
        route=route,
        error_type=error_type,
    )


@pytest.fixture()
def classifier(tmp_path: Any) -> Classifier:
    det_path = build_det_onnx(tmp_path, filename="det.onnx")
    return Classifier.from_path(model_path=det_path)


@pytest.fixture()
def explainer(tmp_path: Any) -> Explainer:
    det_path = build_det_onnx(tmp_path, filename="det.onnx")
    return Explainer.from_path(model_path=det_path)


@pytest.fixture()
def mcd_predictor(tmp_path: Any) -> MCDropoutPredictor:
    mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
    return MCDropoutPredictor.from_path(model_path=mcd_path)


@pytest.fixture()
def client(classifier: Classifier) -> TestClient:
    return TestClient(create_app(predictor=classifier))


class TestErrorTaxonomyOnPredict:
    def test_undecodable_upload_is_counted_as_invalid_image(
        self, client: TestClient
    ) -> None:
        before = _errors(client, "/predict", "invalid_image")
        client.post(
            "/predict", files={"image": ("bad.png", b"not an image", "image/png")}
        )
        assert _errors(client, "/predict", "invalid_image") - before == 1.0

    def test_empty_upload_is_counted_as_empty_file_only(
        self, client: TestClient
    ) -> None:
        before_empty = _errors(client, "/predict", "empty_file")
        before_invalid = _errors(client, "/predict", "invalid_image")
        client.post("/predict", files={"image": ("e.png", b"", "image/png")})
        assert _errors(client, "/predict", "empty_file") - before_empty == 1.0
        assert _errors(client, "/predict", "invalid_image") - before_invalid == 0.0

    def test_missing_file_field_is_counted_as_validation_error(
        self, client: TestClient
    ) -> None:
        before = _errors(client, "/predict", "validation_error")
        assert client.post("/predict").status_code == 400
        assert _errors(client, "/predict", "validation_error") - before == 1.0

    def test_no_predictor_is_counted_as_no_model_loaded(self) -> None:
        client = TestClient(create_app(predictor=None))
        before = _errors(client, "/predict", "no_model_loaded")
        client.post(
            "/predict", files={"image": ("t.png", _make_png_bytes(), "image/png")}
        )
        assert _errors(client, "/predict", "no_model_loaded") - before == 1.0

    def test_successful_upload_records_no_error(self, client: TestClient) -> None:
        client.post(
            "/predict", files={"image": ("t.png", _make_png_bytes(), "image/png")}
        )
        body = client.get("/metrics").text
        for error_type in (
            "invalid_image",
            "empty_file",
            "no_model_loaded",
            "validation_error",
        ):
            assert (
                _sample(
                    body,
                    "inference_errors_total",
                    route="/predict",
                    error_type=error_type,
                )
                == 0.0
            )


class TestInputImageMetricsOnPredict:
    def test_successful_upload_records_size_width_and_height(
        self, client: TestClient
    ) -> None:
        png = _make_png_bytes(width=96, height=48)

        c0, s0 = _hist(client, "inference_input_image_size_bytes")
        w0, ws0 = _hist(client, "inference_input_image_width_pixels")
        h0, hs0 = _hist(client, "inference_input_image_height_pixels")

        client.post("/predict", files={"image": ("t.png", png, "image/png")})

        c1, s1 = _hist(client, "inference_input_image_size_bytes")
        w1, ws1 = _hist(client, "inference_input_image_width_pixels")
        h1, hs1 = _hist(client, "inference_input_image_height_pixels")

        assert c1 - c0 == 1.0
        assert s1 - s0 == pytest.approx(float(len(png)))
        assert w1 - w0 == 1.0
        assert ws1 - ws0 == pytest.approx(96.0)  # not 48 -- width is not transposed
        assert h1 - h0 == 1.0
        assert hs1 - hs0 == pytest.approx(48.0)

    def test_undecodable_upload_records_no_input_observation(
        self, client: TestClient
    ) -> None:
        c0, _ = _hist(client, "inference_input_image_size_bytes")
        w0, _ = _hist(client, "inference_input_image_width_pixels")
        h0, _ = _hist(client, "inference_input_image_height_pixels")
        client.post(
            "/predict", files={"image": ("bad.png", b"not an image", "image/png")}
        )
        c1, _ = _hist(client, "inference_input_image_size_bytes")
        w1, _ = _hist(client, "inference_input_image_width_pixels")
        h1, _ = _hist(client, "inference_input_image_height_pixels")
        assert c1 == c0
        assert w1 == w0
        assert h1 == h0

    def test_empty_upload_records_no_input_observation(
        self, client: TestClient
    ) -> None:
        c0, _ = _hist(client, "inference_input_image_size_bytes")
        client.post("/predict", files={"image": ("e.png", b"", "image/png")})
        c1, _ = _hist(client, "inference_input_image_size_bytes")
        assert c1 == c0

    def test_missing_model_still_records_the_input(self) -> None:
        client = TestClient(create_app(predictor=None))
        png = _make_png_bytes(width=96, height=48)
        c0, s0 = _hist(client, "inference_input_image_size_bytes")
        response = client.post("/predict", files={"image": ("t.png", png, "image/png")})
        assert response.status_code == 503
        c1, s1 = _hist(client, "inference_input_image_size_bytes")
        assert c1 - c0 == 1.0
        assert s1 - s0 == pytest.approx(float(len(png)))


class TestErrorTaxonomyOnExplain:
    @pytest.fixture()
    def client(self, explainer: Explainer) -> TestClient:
        return TestClient(create_app(predictor=explainer))

    def test_undecodable_upload_is_counted_as_invalid_image(
        self, client: TestClient
    ) -> None:
        before = _errors(client, "/explain", "invalid_image")
        client.post(
            "/explain", files={"image": ("bad.png", b"not an image", "image/png")}
        )
        assert _errors(client, "/explain", "invalid_image") - before == 1.0

    def test_empty_upload_is_counted_as_empty_file(self, client: TestClient) -> None:
        before = _errors(client, "/explain", "empty_file")
        client.post("/explain", files={"image": ("e.png", b"", "image/png")})
        assert _errors(client, "/explain", "empty_file") - before == 1.0

    def test_missing_file_field_is_counted_as_validation_error(
        self, client: TestClient
    ) -> None:
        before = _errors(client, "/explain", "validation_error")
        assert client.post("/explain").status_code == 400
        assert _errors(client, "/explain", "validation_error") - before == 1.0


class TestInputImageMetricsOnExplain:
    @pytest.fixture()
    def client(self, explainer: Explainer) -> TestClient:
        return TestClient(create_app(predictor=explainer))

    def test_successful_upload_records_size_width_and_height(
        self, client: TestClient
    ) -> None:
        png = _make_png_bytes(width=96, height=48)

        c0, s0 = _hist(client, "inference_input_image_size_bytes")
        w0, ws0 = _hist(client, "inference_input_image_width_pixels")
        h0, hs0 = _hist(client, "inference_input_image_height_pixels")

        client.post("/explain", files={"image": ("t.png", png, "image/png")})

        c1, s1 = _hist(client, "inference_input_image_size_bytes")
        w1, ws1 = _hist(client, "inference_input_image_width_pixels")
        h1, hs1 = _hist(client, "inference_input_image_height_pixels")

        assert c1 - c0 == 1.0
        assert s1 - s0 == pytest.approx(float(len(png)))
        assert w1 - w0 == 1.0
        assert ws1 - ws0 == pytest.approx(96.0)
        assert h1 - h0 == 1.0
        assert hs1 - hs0 == pytest.approx(48.0)


class TestErrorTaxonomyOnUncertainty:
    @pytest.fixture()
    def client(self, mcd_predictor: MCDropoutPredictor) -> TestClient:
        return TestClient(create_app(predictor=mcd_predictor))

    def test_undecodable_upload_is_counted_as_invalid_image(
        self, client: TestClient
    ) -> None:
        before = _errors(client, "/uncertainty", "invalid_image")
        client.post(
            "/uncertainty", files={"image": ("bad.png", b"not an image", "image/png")}
        )
        assert _errors(client, "/uncertainty", "invalid_image") - before == 1.0

    def test_empty_upload_is_counted_as_empty_file(self, client: TestClient) -> None:
        before = _errors(client, "/uncertainty", "empty_file")
        client.post("/uncertainty", files={"image": ("e.png", b"", "image/png")})
        assert _errors(client, "/uncertainty", "empty_file") - before == 1.0

    def test_missing_file_field_is_counted_as_validation_error(
        self, client: TestClient
    ) -> None:
        before = _errors(client, "/uncertainty", "validation_error")
        assert client.post("/uncertainty").status_code == 400
        assert _errors(client, "/uncertainty", "validation_error") - before == 1.0

    def test_no_predictor_is_counted_as_no_model_loaded(self) -> None:
        client = TestClient(create_app(predictor=None))
        before = _errors(client, "/uncertainty", "no_model_loaded")
        client.post(
            "/uncertainty", files={"image": ("t.png", _make_png_bytes(), "image/png")}
        )
        assert _errors(client, "/uncertainty", "no_model_loaded") - before == 1.0


class TestInputImageMetricsOnUncertainty:
    @pytest.fixture()
    def client(self, mcd_predictor: MCDropoutPredictor) -> TestClient:
        return TestClient(create_app(predictor=mcd_predictor))

    def test_successful_upload_records_size_width_and_height(
        self, client: TestClient
    ) -> None:
        png = _make_png_bytes(width=96, height=48)

        c0, s0 = _hist(client, "inference_input_image_size_bytes")
        w0, ws0 = _hist(client, "inference_input_image_width_pixels")
        h0, hs0 = _hist(client, "inference_input_image_height_pixels")

        client.post("/uncertainty", files={"image": ("t.png", png, "image/png")})

        c1, s1 = _hist(client, "inference_input_image_size_bytes")
        w1, ws1 = _hist(client, "inference_input_image_width_pixels")
        h1, hs1 = _hist(client, "inference_input_image_height_pixels")

        assert c1 - c0 == 1.0
        assert s1 - s0 == pytest.approx(float(len(png)))
        assert w1 - w0 == 1.0
        assert ws1 - ws0 == pytest.approx(96.0)
        assert h1 - h0 == 1.0
        assert hs1 - hs0 == pytest.approx(48.0)


class TestReadinessProbeDoesNotInflateErrorBudget:
    def test_readiness_probe_records_no_error(self) -> None:
        client = TestClient(create_app(predictor=None))
        assert client.get("/readyz").status_code == 503
        body = client.get("/metrics").text
        for error_type in (
            "invalid_image",
            "empty_file",
            "no_model_loaded",
            "validation_error",
        ):
            assert (
                _sample(
                    body,
                    "inference_errors_total",
                    route="/readyz",
                    error_type=error_type,
                )
                == 0.0
            )
