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

"""Prometheus instrumentation for the inference serving layer.

One :class:`Metrics` instance is built per FastAPI application by
:func:`build_metrics`, each owning a private ``CollectorRegistry``.  Every
observer degrades to a no-op when the optional ``prometheus_client``
dependency is absent, so callers never guard their call sites.
"""

from typing import Any, Dict, FrozenSet, Optional, Tuple

from radiologist.inference.optional import _prometheus_client  # noqa: F401

# --- closed label sets (real values, not stubs) ---------------------------

ROUTE_LABELS: FrozenSet[str] = frozenset(
    {"/predict", "/explain", "/uncertainty", "/healthz", "/readyz", "/metrics"}
)
UNMATCHED_ROUTE: str = "unmatched"
ERROR_TYPES: FrozenSet[str] = frozenset(
    {"invalid_image", "empty_file", "no_model_loaded", "validation_error"}
)

# --- histogram buckets (real values, not stubs) ---------------------------

_INF: float = float("inf")

DURATION_BUCKETS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    _INF,
)
IMAGE_SIZE_BUCKETS: Tuple[float, ...] = (
    1e3,
    1e4,
    5e4,
    1e5,
    2.5e5,
    5e5,
    1e6,
    2.5e6,
    5e6,
    1e7,
    _INF,
)
IMAGE_DIM_BUCKETS: Tuple[float, ...] = (
    64,
    128,
    224,
    256,
    384,
    512,
    768,
    1024,
    2048,
    4096,
    _INF,
)
CONFIDENCE_BUCKETS: Tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    _INF,
)
ENTROPY_BUCKETS: Tuple[float, ...] = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.693,
    0.8,
    1.0,
    1.5,
    2.0,
    _INF,
)
STD_BUCKETS: Tuple[float, ...] = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    _INF,
)

_PLAIN_TEXT: str = "text/plain; charset=utf-8"


class Metrics:
    """Per-application Prometheus metric catalogue and recording surface.

    Args:
        client: The imported ``prometheus_client`` module, or ``None`` when the
            optional dependency is unavailable.  When ``None`` every recording
            method is a no-op and :meth:`render_latest` returns an empty
            payload, so call sites never branch on availability.
    """

    def __init__(self, client: Optional[Any]) -> None:
        """Build the metric catalogue, or disable it, for one application.

        When ``client`` is ``None`` sets an internal disabled flag and builds
        nothing. Otherwise creates a fresh ``client.CollectorRegistry()``
        owned solely by this instance and registers all ten metric families
        onto it. Never touches the global default registry, so constructing
        many instances in one process can never raise "Duplicated timeseries
        in CollectorRegistry".

        Args:
            client: The imported ``prometheus_client`` module, or ``None``.
        """
        if client is None:
            self._enabled = False
            return
        self._enabled = True

        self._registry = client.CollectorRegistry()

        self._requests_total = client.Counter(
            "inference_requests_total",
            "Total number of inference API requests.",
            ("route", "status"),
            registry=self._registry,
        )
        self._request_duration_seconds = client.Histogram(
            "inference_request_duration_seconds",
            "Wall-clock duration of inference API requests, in seconds.",
            ("route",),
            buckets=DURATION_BUCKETS,
            registry=self._registry,
        )
        self._requests_in_progress = client.Gauge(
            "inference_requests_in_progress",
            "Number of inference API requests currently being served.",
            ("route",),
            registry=self._registry,
        )
        self._errors_total = client.Counter(
            "inference_errors_total",
            "Total number of request-level inference errors.",
            ("route", "error_type"),
            registry=self._registry,
        )
        self._input_image_size_bytes = client.Histogram(
            "inference_input_image_size_bytes",
            "Size of uploaded input images, in bytes.",
            buckets=IMAGE_SIZE_BUCKETS,
            registry=self._registry,
        )
        self._input_image_width_pixels = client.Histogram(
            "inference_input_image_width_pixels",
            "Pre-resize width of uploaded input images, in pixels.",
            buckets=IMAGE_DIM_BUCKETS,
            registry=self._registry,
        )
        self._input_image_height_pixels = client.Histogram(
            "inference_input_image_height_pixels",
            "Pre-resize height of uploaded input images, in pixels.",
            buckets=IMAGE_DIM_BUCKETS,
            registry=self._registry,
        )
        self._predicted_class_total = client.Counter(
            "inference_predicted_class_total",
            "Total number of predictions per predicted class.",
            ("class",),
            registry=self._registry,
        )
        self._confidence = client.Histogram(
            "inference_confidence",
            "Maximum predicted class probability of a prediction.",
            buckets=CONFIDENCE_BUCKETS,
            registry=self._registry,
        )
        self._predictive_entropy = client.Histogram(
            "inference_predictive_entropy",
            "Predictive entropy of the mean MC-Dropout prediction.",
            buckets=ENTROPY_BUCKETS,
            registry=self._registry,
        )
        self._uncertainty_std_max = client.Histogram(
            "inference_uncertainty_std_max",
            "Maximum per-class standard deviation across MC-Dropout passes.",
            buckets=STD_BUCKETS,
            registry=self._registry,
        )
        self._client = client

    @property
    def enabled(self) -> bool:
        """Report whether this instance was built with a real client.

        Returns:
            ``True`` iff this instance was built with a real
            ``prometheus_client`` module. Read by ``_build_app`` to decide
            whether to register the HTTP middleware and the ``GET /metrics``
            route.
        """
        return self._enabled

    def route_label(self, path: str) -> str:
        """Normalize a request path to a bounded-cardinality label.

        Args:
            path: The raw request path.

        Returns:
            ``path`` when it is a member of ``ROUTE_LABELS``, else
            ``UNMATCHED_ROUTE``. Never returns caller-controlled text, so
            label cardinality stays bounded. Pure: returns a value even when
            disabled.
        """
        return path if path in ROUTE_LABELS else UNMATCHED_ROUTE

    def track_request_start(self, route: str) -> None:
        """Record that a request has started.

        Increments ``inference_requests_in_progress{route}``. No-op when
        disabled.

        Args:
            route: The bounded route label, from :meth:`route_label`.
        """
        if not self._enabled:
            return
        self._requests_in_progress.labels(route=route).inc()

    def track_request_end(
        self, route: str, status: int, duration_seconds: float
    ) -> None:
        """Record that a request has finished.

        Decrements ``inference_requests_in_progress{route}``, increments
        ``inference_requests_total{route, status=str(status)}`` by one, and
        observes ``duration_seconds`` into
        ``inference_request_duration_seconds{route}``. No-op when disabled.
        Must be safe to call from a ``finally`` block after a failed
        dispatch.

        Args:
            route: The bounded route label, from :meth:`route_label`.
            status: The HTTP status code of the completed response.
            duration_seconds: Wall-clock duration of the request.
        """
        if not self._enabled:
            return
        self._requests_in_progress.labels(route=route).dec()
        self._requests_total.labels(route=route, status=str(status)).inc()
        self._request_duration_seconds.labels(route=route).observe(duration_seconds)

    def observe_error(self, route: str, error_type: str) -> None:
        """Record a request-level error.

        Increments ``inference_errors_total{route, error_type}`` by one.
        ``error_type`` must be a member of ``ERROR_TYPES``; an unknown value
        is silently ignored rather than raised, so instrumentation can never
        break a request. No-op when disabled.

        Args:
            route: The bounded route label, from :meth:`route_label`.
            error_type: One of the closed ``ERROR_TYPES`` values.
        """
        if not self._enabled or error_type not in ERROR_TYPES:
            return
        self._errors_total.labels(route=route, error_type=error_type).inc()

    def observe_input_image(self, size_bytes: int, width: int, height: int) -> None:
        """Record size and dimensions of an uploaded input image.

        Observes one sample into each of
        ``inference_input_image_size_bytes``,
        ``inference_input_image_width_pixels`` and
        ``inference_input_image_height_pixels``. ``width``/``height`` are the
        pre-resize dimensions of the uploaded file. No-op when disabled.

        Args:
            size_bytes: Size of the uploaded file, in bytes.
            width: Pre-resize width, in pixels.
            height: Pre-resize height, in pixels.
        """
        if not self._enabled:
            return
        self._input_image_size_bytes.observe(size_bytes)
        self._input_image_width_pixels.observe(width)
        self._input_image_height_pixels.observe(height)

    def observe_predicted_class(self, predicted_class: str) -> None:
        """Record the predicted class of a request.

        Increments ``inference_predicted_class_total{class}`` by one for
        ``predicted_class``. Unlabelled by route. No-op when disabled.

        Args:
            predicted_class: The predicted class label.
        """
        if not self._enabled:
            return
        self._predicted_class_total.labels(**{"class": predicted_class}).inc()

    def observe_confidence(self, probabilities: Dict[str, float]) -> None:
        """Record the confidence of a prediction.

        Observes ``max(probabilities.values())`` into
        ``inference_confidence``. Observes nothing when ``probabilities`` is
        empty. No-op when disabled.

        Args:
            probabilities: Per-class predicted probabilities.
        """
        if not self._enabled or not probabilities:
            return
        self._confidence.observe(max(probabilities.values()))

    def observe_uncertainty(
        self, predictive_entropy: float, std_per_class: Dict[str, float]
    ) -> None:
        """Record predictive uncertainty of an MC-Dropout prediction.

        Observes ``predictive_entropy`` into
        ``inference_predictive_entropy``, and
        ``max(std_per_class.values())`` into
        ``inference_uncertainty_std_max``. The std observation is skipped
        when ``std_per_class`` is empty; the entropy observation is not.
        No-op when disabled.

        Args:
            predictive_entropy: Predictive entropy of the mean prediction.
            std_per_class: Per-class standard deviation across MC passes.
        """
        if not self._enabled:
            return
        self._predictive_entropy.observe(predictive_entropy)
        if std_per_class:
            self._uncertainty_std_max.observe(max(std_per_class.values()))

    def render_latest(self) -> Tuple[bytes, str]:
        """Render the current scrape payload for this instance's registry.

        Returns:
            A tuple of ``(exposition payload, content type)``, i.e.
            ``(client.generate_latest(registry), client.CONTENT_TYPE_LATEST)``.
            Returns ``(b"", _PLAIN_TEXT)`` when disabled.
        """
        if not self._enabled:
            return b"", _PLAIN_TEXT
        return (
            self._client.generate_latest(self._registry),
            self._client.CONTENT_TYPE_LATEST,
        )


def build_metrics() -> Metrics:
    """Build the metrics recorder for one FastAPI application.

    Reads the ``_prometheus_client`` sentinel at call time (never at import
    time) so tests may disable instrumentation with
    ``monkeypatch.setattr`` before building an application.

    Returns:
        A :class:`Metrics` bound to a freshly created private registry when
        ``prometheus_client`` is importable, otherwise a no-op
        :class:`Metrics`.
    """
    return Metrics(_prometheus_client)
