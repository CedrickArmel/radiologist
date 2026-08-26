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

"""Behavioural tests for the GET /metrics scrape endpoint.

Covers scraping the Prometheus text exposition, per-application isolation of
the underlying registry, and graceful degradation (404) when the optional
``prometheus_client`` dependency is absent.
"""

from typing import Any

import pytest
from _helpers import _make_png_bytes, _sample, build_det_onnx
from fastapi.testclient import TestClient

from radiologist.inference import Classifier, create_app


@pytest.fixture()
def classifier(tmp_path: Any) -> Classifier:
    det_path = build_det_onnx(tmp_path, filename="det.onnx")
    return Classifier.from_path(model_path=det_path)


@pytest.fixture()
def client(classifier: Classifier) -> TestClient:
    return TestClient(create_app(predictor=classifier))


class TestScrapeExposition:
    def test_scrape_returns_prometheus_exposition(self, client: TestClient) -> None:
        import prometheus_client

        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == prometheus_client.CONTENT_TYPE_LATEST
        assert "# TYPE inference_requests_total counter" in response.text

    def test_all_ten_families_declared_before_any_traffic(
        self, client: TestClient
    ) -> None:
        body = client.get("/metrics").text
        expected = (
            "inference_requests_total",
            "inference_request_duration_seconds",
            "inference_requests_in_progress",
            "inference_errors_total",
            "inference_input_image_size_bytes",
            "inference_input_image_width_pixels",
            "inference_input_image_height_pixels",
            "inference_predicted_class_total",
            "inference_confidence",
            "inference_predictive_entropy",
            "inference_uncertainty_std_max",
        )
        for name in expected:
            assert f"# TYPE {name}" in body, f"missing declaration for {name}"


class TestPerApplicationIsolation:
    def test_two_apps_report_independently(self, classifier: Classifier) -> None:
        a = TestClient(create_app(predictor=classifier))
        b = TestClient(create_app(predictor=classifier))
        a.post(
            "/predict",
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
        counted = _sample(
            b.get("/metrics").text,
            "inference_requests_total",
            route="/predict",
            status="200",
        )
        assert counted == 0.0

    def test_many_apps_can_be_built_without_duplicate_timeseries_error(
        self, classifier: Classifier
    ) -> None:
        for _ in range(5):
            c = TestClient(create_app(predictor=classifier))
            response = c.get("/metrics")
            assert response.status_code == 200


class TestMetricsAbsentWhenExtraMissing:
    def test_metrics_route_404_and_other_routes_unaffected(
        self, monkeypatch: Any, classifier: Classifier
    ) -> None:
        import radiologist.inference.metrics as metrics_module

        monkeypatch.setattr(metrics_module, "_prometheus_client", None)
        client = TestClient(create_app(predictor=classifier))
        assert client.get("/metrics").status_code == 404
        ok = client.post(
            "/predict",
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
        assert ok.status_code == 200
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
