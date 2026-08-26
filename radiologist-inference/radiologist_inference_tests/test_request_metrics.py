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

"""Behavioural tests for the RED request metrics (1-3).

Covers ``inference_requests_total``, ``inference_request_duration_seconds``
and ``inference_requests_in_progress``, driven exclusively through real HTTP
traffic against a real ``TestClient``. Assertions compare scrape deltas
(scrape -> act -> scrape -> compare), never absolute values.
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


class TestRequestsTotal:
    def test_successful_prediction_is_counted_for_its_route(
        self, client: TestClient
    ) -> None:
        before = _sample(
            client.get("/metrics").text,
            "inference_requests_total",
            route="/predict",
            status="200",
        )
        client.post(
            "/predict",
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
        after = _sample(
            client.get("/metrics").text,
            "inference_requests_total",
            route="/predict",
            status="200",
        )
        assert after - before == 1.0

    def test_failed_prediction_with_no_model_is_counted_as_503(self) -> None:
        client = TestClient(create_app(predictor=None))
        before = _sample(
            client.get("/metrics").text,
            "inference_requests_total",
            route="/predict",
            status="503",
        )
        client.post(
            "/predict",
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
        after = _sample(
            client.get("/metrics").text,
            "inference_requests_total",
            route="/predict",
            status="503",
        )
        assert after - before == 1.0

    def test_unknown_path_is_counted_as_unmatched(self, client: TestClient) -> None:
        client.get("/this-route-does-not-exist")
        body = client.get("/metrics").text
        assert (
            _sample(body, "inference_requests_total", route="unmatched", status="404")
            >= 1.0
        )

    def test_healthz_and_readyz_are_counted_like_any_other_route(
        self, client: TestClient
    ) -> None:
        before_health = _sample(
            client.get("/metrics").text,
            "inference_requests_total",
            route="/healthz",
            status="200",
        )
        before_ready = _sample(
            client.get("/metrics").text,
            "inference_requests_total",
            route="/readyz",
            status="200",
        )
        client.get("/healthz")
        client.get("/readyz")
        body = client.get("/metrics").text
        after_health = _sample(
            body, "inference_requests_total", route="/healthz", status="200"
        )
        after_ready = _sample(
            body, "inference_requests_total", route="/readyz", status="200"
        )
        assert after_health - before_health == 1.0
        assert after_ready - before_ready == 1.0


class TestScrapeDoesNotCountItself:
    def test_scraping_does_not_count_itself(self, client: TestClient) -> None:
        client.get("/metrics")
        first = client.get("/metrics").text
        second = client.get("/metrics").text
        for status in ("200", "404", "503"):
            assert (
                _sample(
                    first, "inference_requests_total", route="/metrics", status=status
                )
                == 0.0
            )
            assert (
                _sample(
                    second,
                    "inference_requests_total",
                    route="/metrics",
                    status=status,
                )
                == 0.0
            )

    def test_repeated_scrapes_do_not_move_duration_or_in_progress(
        self, client: TestClient
    ) -> None:
        client.get("/metrics")
        first = client.get("/metrics").text
        second = client.get("/metrics").text
        for body in (first, second):
            assert (
                _sample(
                    body,
                    "inference_request_duration_seconds_count",
                    route="/metrics",
                )
                == 0.0
            )
            assert (
                _sample(body, "inference_requests_in_progress", route="/metrics") == 0.0
            )


class TestRequestDuration:
    def test_successful_prediction_adds_one_observation_with_positive_duration(
        self, client: TestClient
    ) -> None:
        before_count = _sample(
            client.get("/metrics").text,
            "inference_request_duration_seconds_count",
            route="/predict",
        )
        before_sum = _sample(
            client.get("/metrics").text,
            "inference_request_duration_seconds_sum",
            route="/predict",
        )
        client.post(
            "/predict",
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
        body = client.get("/metrics").text
        after_count = _sample(
            body, "inference_request_duration_seconds_count", route="/predict"
        )
        after_sum = _sample(
            body, "inference_request_duration_seconds_sum", route="/predict"
        )
        assert after_count - before_count == 1.0
        assert after_sum - before_sum > 0.0


class TestRequestsInProgress:
    def test_gauge_is_zero_between_requests(self, client: TestClient) -> None:
        body = client.get("/metrics").text
        for route in ("/predict", "/healthz", "/readyz", "/explain", "/uncertainty"):
            assert _sample(body, "inference_requests_in_progress", route=route) == 0.0

    def test_gauge_is_released_after_a_failed_request(self) -> None:
        client = TestClient(create_app(predictor=None))
        client.post(
            "/predict",
            files={"image": ("t.png", _make_png_bytes(), "image/png")},
        )
        body = client.get("/metrics").text
        assert _sample(body, "inference_requests_in_progress", route="/predict") == 0.0
