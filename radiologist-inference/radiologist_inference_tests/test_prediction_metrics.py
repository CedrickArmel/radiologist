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

"""Behavioural tests for the prediction, confidence and uncertainty metrics.

Covers ``inference_predicted_class_total``, ``inference_confidence``,
``inference_predictive_entropy`` and ``inference_uncertainty_std_max``, driven
exclusively through real HTTP traffic against a real ``TestClient``.
Assertions compare scrape deltas (scrape -> act -> scrape -> compare), and
values are always read from the JSON response body of the triggering
request rather than hard-coded, since the tiny ONNX fixtures (and MC-Dropout
in particular) do not produce deterministic outputs.
"""

from typing import Any

import pytest
from _helpers import _hist, _make_png_bytes, _sample, build_det_onnx, build_mcd_onnx
from fastapi.testclient import TestClient

from radiologist.inference import Classifier, Explainer, MCDropoutPredictor, create_app


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


def test_prediction_records_its_class_and_confidence(client: TestClient) -> None:
    c0, s0 = _hist(client, "inference_confidence")
    response = client.post(
        "/predict", files={"image": ("t.png", _make_png_bytes(), "image/png")}
    )
    body = response.json()
    expected_class = body["predicted_class"]
    expected_conf = max(body["probabilities"].values())

    exposition = client.get("/metrics").text
    counted = _sample(
        exposition, "inference_predicted_class_total", **{"class": expected_class}
    )
    c1, s1 = _hist(client, "inference_confidence")

    assert counted == 1.0
    assert c1 - c0 == 1.0
    assert s1 - s0 == pytest.approx(expected_conf)


def test_repeated_predictions_of_the_same_class_accumulate(
    client: TestClient,
) -> None:
    body = client.post(
        "/predict", files={"image": ("t.png", _make_png_bytes(), "image/png")}
    ).json()
    expected_class = body["predicted_class"]
    before = _sample(
        client.get("/metrics").text,
        "inference_predicted_class_total",
        **{"class": expected_class},
    )
    client.post("/predict", files={"image": ("t.png", _make_png_bytes(), "image/png")})
    after = _sample(
        client.get("/metrics").text,
        "inference_predicted_class_total",
        **{"class": expected_class},
    )
    assert after - before == 1.0


def test_explanation_records_its_class_but_no_confidence(
    explainer: Explainer,
) -> None:
    client = TestClient(create_app(predictor=explainer))
    c0, _ = _hist(client, "inference_confidence")
    body = client.post(
        "/explain", files={"image": ("t.png", _make_png_bytes(), "image/png")}
    ).json()

    exposition = client.get("/metrics").text
    counted = _sample(
        exposition,
        "inference_predicted_class_total",
        **{"class": body["predicted_class"]},
    )
    c1, _ = _hist(client, "inference_confidence")

    assert counted == 1.0
    assert c1 == c0  # explanation carries no probabilities


def test_uncertainty_records_entropy_and_max_std(
    mcd_predictor: MCDropoutPredictor,
) -> None:
    client = TestClient(create_app(predictor=mcd_predictor))
    e0, es0 = _hist(client, "inference_predictive_entropy")
    d0, ds0 = _hist(client, "inference_uncertainty_std_max")

    body = client.post(
        "/uncertainty", files={"image": ("t.png", _make_png_bytes(), "image/png")}
    ).json()
    expected_entropy = body["predictive_entropy"]
    expected_std_max = max(body["std_per_class"].values())

    e1, es1 = _hist(client, "inference_predictive_entropy")
    d1, ds1 = _hist(client, "inference_uncertainty_std_max")

    assert e1 - e0 == 1.0
    assert es1 - es0 == pytest.approx(expected_entropy)
    assert d1 - d0 == 1.0
    assert ds1 - ds0 == pytest.approx(expected_std_max)
    # max, not sum: a differing per-class spread records only the largest
    assert ds1 - ds0 != pytest.approx(sum(body["std_per_class"].values()))


def test_uncertainty_records_no_class_and_no_confidence(
    mcd_predictor: MCDropoutPredictor,
) -> None:
    client = TestClient(create_app(predictor=mcd_predictor))
    c0, _ = _hist(client, "inference_confidence")
    client.post(
        "/uncertainty", files={"image": ("t.png", _make_png_bytes(), "image/png")}
    )
    exposition = client.get("/metrics").text
    c1, _ = _hist(client, "inference_confidence")

    assert c1 == c0
    assert "inference_predicted_class_total{" not in exposition


def test_predict_and_explain_add_no_entropy_or_std_observation(
    client: TestClient,
) -> None:
    e0, _ = _hist(client, "inference_predictive_entropy")
    d0, _ = _hist(client, "inference_uncertainty_std_max")
    client.post("/predict", files={"image": ("t.png", _make_png_bytes(), "image/png")})
    e1, _ = _hist(client, "inference_predictive_entropy")
    d1, _ = _hist(client, "inference_uncertainty_std_max")
    assert e1 == e0
    assert d1 == d0


def test_rejected_request_records_nothing(client: TestClient) -> None:
    c0, _ = _hist(client, "inference_confidence")
    client.post("/predict", files={"image": ("bad.png", b"nope", "image/png")})
    client.post("/predict", files={"image": ("e.png", b"", "image/png")})
    exposition = client.get("/metrics").text
    c1, _ = _hist(client, "inference_confidence")

    assert c1 == c0
    assert "inference_predicted_class_total{" not in exposition
