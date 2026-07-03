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

"""Behavioural tests for the isinstance-driven create_app() factory."""

import io
from typing import Any

import numpy as np
import pytest
from _helpers import build_det_onnx, build_mcd_onnx
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from radiologist.inference import Classifier, Explainer, MCDropoutPredictor, create_app


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def classifier(tmp_path) -> Classifier:
    det_path = build_det_onnx(tmp_path, filename="det.onnx")
    return Classifier.from_path(det_path=det_path)


@pytest.fixture()
def explainer(tmp_path) -> Explainer:
    det_path = build_det_onnx(tmp_path, filename="det.onnx")
    return Explainer.from_path(det_path=det_path)


@pytest.fixture()
def mcd_predictor(tmp_path) -> MCDropoutPredictor:
    det_path = build_det_onnx(tmp_path, filename="det.onnx")
    mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
    return MCDropoutPredictor.from_path(det_path=det_path, mcd_path=mcd_path)


# ---------------------------------------------------------------------------
# AC: create_app(Classifier(...)) serves /predict + /healthz, 404s the rest.
# ---------------------------------------------------------------------------


class TestClassifierApp:
    @pytest.fixture()
    def client(self, classifier: Classifier) -> TestClient:
        return TestClient(create_app(predictor=classifier))

    def test_predict_returns_200_with_probabilities(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body["probabilities"].keys()) == {"NORMAL", "ABNORMAL"}
        assert "predicted_class" in body

    def test_explain_route_is_absent(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 404

    def test_uncertainty_route_is_absent(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 404

    def test_healthz_returns_200(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_readyz_returns_200(self, client: TestClient) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_predict_returns_400_on_missing_image(self, client: TestClient) -> None:
        response = client.post("/predict")
        assert response.status_code == 400

    def test_predict_returns_400_on_invalid_image_bytes(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/predict",
            files={"image": ("bad.png", b"not an image", "image/png")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# AC: create_app(Explainer(...)) additionally serves /explain.
# ---------------------------------------------------------------------------


class TestExplainerApp:
    @pytest.fixture()
    def client(self, explainer: Explainer) -> TestClient:
        return TestClient(create_app(predictor=explainer))

    def test_predict_still_served(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200

    def test_explain_returns_200_with_saliency_map(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["saliency_map"], list)
        assert len(body["saliency_map"]) > 0
        assert "predicted_class" in body

    def test_uncertainty_route_is_absent(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# AC: create_app(MCDropoutPredictor(...)) serves /uncertainty, 404s /predict.
# ---------------------------------------------------------------------------


class TestMCDropoutApp:
    @pytest.fixture()
    def client(self, mcd_predictor: MCDropoutPredictor) -> TestClient:
        return TestClient(create_app(predictor=mcd_predictor))

    def test_uncertainty_returns_200_with_stats(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert "mean_probabilities" in body
        assert "std_per_class" in body
        assert isinstance(body["predictive_entropy"], float)

    def test_predict_route_is_absent(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 404

    def test_explain_route_is_absent(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 404

    def test_healthz_returns_200(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_readyz_returns_200(self, client: TestClient) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# AC: a wired route with no predictor injected returns 503.
# ---------------------------------------------------------------------------


class TestNoPredictorInjected:
    @pytest.fixture()
    def client(self) -> TestClient:
        return TestClient(create_app(predictor=None))

    def test_predict_returns_503(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/predict", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 503

    def test_explain_returns_503(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/explain", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 503

    def test_uncertainty_returns_503(self, client: TestClient) -> None:
        png = _make_png_bytes()
        response = client.post(
            "/uncertainty", files={"image": ("test.png", png, "image/png")}
        )
        assert response.status_code == 503

    def test_healthz_returns_200(self, client: TestClient) -> None:
        """healthz is pure liveness: 200 even with no predictor loaded."""
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_readyz_returns_503(self, client: TestClient) -> None:
        """readyz owns readiness: 503 when no predictor is loaded."""
        response = client.get("/readyz")
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# AC: create_app raises RuntimeError naming 'serve' when fastapi is absent.
# ---------------------------------------------------------------------------


class TestCreateAppRuntimeErrorWhenFastapiAbsent:
    def test_create_app_raises_runtime_error_naming_serve_extra(
        self, monkeypatch: Any
    ) -> None:
        import radiologist.inference.app as app_module

        monkeypatch.setattr(app_module, "_fastapi", None)
        with pytest.raises(RuntimeError, match="serve"):
            app_module.create_app()
