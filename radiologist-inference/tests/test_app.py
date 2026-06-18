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

"""Behavioural tests for create_app() and all HTTP routes (issue #81)."""

import io
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from radiologist.inference import Prediction, UncertaintyResult, create_app
from radiologist.inference._stubs import Explanation


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def mock_predictor() -> MagicMock:
    p = MagicMock()
    p.predict.return_value = Prediction(
        probabilities={"NORMAL": 0.8, "ABNORMAL": 0.2},
        predicted_class="NORMAL",
    )
    saliency = np.zeros((64, 64), dtype=np.float32)
    p.explain.return_value = Explanation(
        saliency_map=saliency, predicted_class="NORMAL"
    )
    p.predict_with_uncertainty.return_value = UncertaintyResult(
        mean_probabilities={"NORMAL": 0.7, "ABNORMAL": 0.3},
        std_per_class={"NORMAL": 0.05, "ABNORMAL": 0.05},
        predictive_entropy=0.42,
        n_passes=30,
    )
    return p


@pytest.fixture()
def client(mock_predictor: MagicMock) -> TestClient:
    app = create_app(predictor=mock_predictor)
    return TestClient(app)


@pytest.fixture()
def client_no_model() -> TestClient:
    app = create_app(predictor=None)
    return TestClient(app)


# ---------------------------------------------------------------------------
# AC: POST /predict — 200 with probabilities
# ---------------------------------------------------------------------------


class TestPostPredict:
    def test_returns_200_with_class_probabilities(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert "probabilities" in body
        assert isinstance(body["probabilities"], dict)

    def test_probabilities_keys_are_class_names(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        body = response.json()
        assert set(body["probabilities"].keys()) == {"NORMAL", "ABNORMAL"}

    def test_predicted_class_present_in_response(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        body = response.json()
        assert "predicted_class" in body


# ---------------------------------------------------------------------------
# AC: POST /explain — 200 with saliency map
# ---------------------------------------------------------------------------


class TestPostExplain:
    def test_returns_200_with_saliency_map(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert "saliency_map" in body
        assert isinstance(body["saliency_map"], list)

    def test_saliency_map_is_nested_list(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        body = response.json()
        assert len(body["saliency_map"]) > 0

    def test_predicted_class_present(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        body = response.json()
        assert "predicted_class" in body


# ---------------------------------------------------------------------------
# AC: POST /uncertainty — 200 with spread and entropy
# ---------------------------------------------------------------------------


class TestPostUncertainty:
    def test_returns_200_with_per_class_spread(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert "std_per_class" in body
        assert isinstance(body["std_per_class"], dict)

    def test_returns_predictive_entropy(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        body = response.json()
        assert "predictive_entropy" in body
        assert isinstance(body["predictive_entropy"], float)

    def test_returns_mean_probabilities(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        body = response.json()
        assert "mean_probabilities" in body


# ---------------------------------------------------------------------------
# AC: GET /healthz — 200 when model loaded
# ---------------------------------------------------------------------------


class TestGetHealthz:
    def test_returns_200_when_model_is_loaded(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_healthz_body_contains_status_ok(self, client: TestClient) -> None:
        response = client.get("/healthz")
        body = response.json()
        assert body.get("status") == "ok"


# ---------------------------------------------------------------------------
# AC: POST /predict — 400 on malformed/missing image
# ---------------------------------------------------------------------------


class TestPostPredictBadInput:
    def test_returns_400_when_image_field_missing(self, client: TestClient) -> None:
        response = client.post("/predict")
        assert response.status_code == 400

    def test_returns_400_when_image_bytes_are_not_valid_image(
        self, client: TestClient
    ) -> None:
        garbage = b"this is not an image"
        response = client.post(
            "/predict",
            files={"image": ("bad.png", garbage, "image/png")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# AC: inference endpoints 503 when no model loaded
# ---------------------------------------------------------------------------


class TestNoModelLoaded:
    def test_predict_returns_503_when_no_predictor(
        self, client_no_model: TestClient
    ) -> None:
        png = _make_png_bytes()
        response = client_no_model.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 503

    def test_explain_returns_503_when_no_predictor(
        self, client_no_model: TestClient
    ) -> None:
        png = _make_png_bytes()
        response = client_no_model.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 503

    def test_uncertainty_returns_503_when_no_predictor(
        self, client_no_model: TestClient
    ) -> None:
        png = _make_png_bytes()
        response = client_no_model.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# AC: create_app raises RuntimeError when serve extra absent (already tested
#     in test_public_api.py — verified here for completeness)
# ---------------------------------------------------------------------------


class TestCreateAppRuntimeErrorWhenFastapiAbsent:
    def test_create_app_raises_runtime_error_naming_serve_extra(
        self, monkeypatch: Any
    ) -> None:
        import radiologist.inference._stubs as stubs

        monkeypatch.setattr(stubs, "_fastapi", None)
        with pytest.raises(RuntimeError, match="serve"):
            stubs.create_app()
