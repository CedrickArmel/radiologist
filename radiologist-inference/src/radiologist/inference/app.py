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

"""FastAPI application factory for the radiologist inference serving layer."""

import io
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from radiologist.inference.optional import _fastapi

if TYPE_CHECKING:
    from radiologist.inference.base_predictor import BasePredictor


def create_app(predictor: Optional["BasePredictor"] = None) -> Any:
    """Create and return the FastAPI application instance.

    Args:
        predictor: Optional predictor instance to inject at startup.

    Returns:
        Configured FastAPI application instance.

    Raises:
        RuntimeError: When the ``serve`` extra (fastapi, uvicorn) is not
            installed.
    """
    if _fastapi is None:
        raise RuntimeError(
            "The 'serve' extra is required to use create_app. "
            "Install it with: pip install radiologist-inference[serve]"
        )
    return _build_app(_fastapi, predictor)


def _build_app(fastapi_mod: Any, predictor: Optional[Any]) -> Any:  # noqa: C901
    """Build and return the FastAPI app with capability-matched routes.

    Routes are wired based on ``isinstance`` checks against the predictor:
    ``Classifier`` (and its subclass ``Explainer``) gets ``/predict``,
    ``Explainer`` additionally gets ``/explain``, and ``MCDropoutPredictor``
    gets ``/uncertainty``. When ``predictor`` is ``None`` the type is unknown,
    so every route is wired and each falls back to the 503 "no model loaded"
    guard until a predictor is injected. ``/healthz`` (pure liveness) and
    ``/readyz`` (readiness, 503 until a predictor is loaded) are always wired.

    Args:
        fastapi_mod: The imported fastapi module (passed to avoid re-importing).
        predictor: Optional predictor instance injected at startup.

    Returns:
        Configured FastAPI application.
    """
    import fastapi  # type: ignore[import-untyped]
    from fastapi.exceptions import (
        RequestValidationError,  # type: ignore[import-untyped]
    )
    from starlette.requests import Request  # type: ignore[import-untyped]
    from starlette.responses import JSONResponse  # type: ignore[import-untyped]

    from radiologist.inference.classifier import Classifier
    from radiologist.inference.explainer import Explainer
    from radiologist.inference.mc_dropout import MCDropoutPredictor

    app = fastapi_mod.FastAPI(title="Radiologist Inference API")
    state_holder: Dict[str, Any] = {"predictor": predictor}

    HTTPException = fastapi_mod.HTTPException

    wire_predict = predictor is None or isinstance(predictor, Classifier)
    wire_explain = predictor is None or isinstance(predictor, Explainer)
    wire_uncertainty = predictor is None or isinstance(predictor, MCDropoutPredictor)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    def _get_predictor(route: Optional[str] = None) -> Any:
        p = state_holder["predictor"]
        if p is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Supply a predictor instance at startup.",
            )
        return p

    def _load_pil(data: bytes, route: Optional[str] = None) -> PILImage.Image:
        try:
            return PILImage.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid image: {exc}"
            ) from exc

    async def _handle(image: Any, method_name: str, route: str) -> Any:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty image file.")
        pil_img = _load_pil(raw, route)
        p = _get_predictor(route)
        return getattr(p, method_name)(pil_img)

    if wire_predict:

        @app.post("/predict")
        async def predict(
            image: fastapi.UploadFile = fastapi.File(...),
        ) -> Dict[str, Any]:
            result = await _handle(image, "predict", "/predict")
            return {
                "probabilities": result.probabilities,
                "predicted_class": result.predicted_class,
            }

    if wire_explain:

        @app.post("/explain")
        async def explain(
            image: fastapi.UploadFile = fastapi.File(...),
        ) -> Dict[str, Any]:
            result = await _handle(image, "explain", "/explain")
            saliency: List[List[float]] = result.saliency_map.tolist()
            return {
                "saliency_map": saliency,
                "predicted_class": result.predicted_class,
            }

    if wire_uncertainty:

        @app.post("/uncertainty")
        async def uncertainty(
            image: fastapi.UploadFile = fastapi.File(...),
        ) -> Dict[str, Any]:
            result = await _handle(image, "predict_with_uncertainty", "/uncertainty")
            return {
                "mean_probabilities": result.mean_probabilities,
                "std_per_class": result.std_per_class,
                "predictive_entropy": result.predictive_entropy,
                "n_passes": result.n_passes,
            }

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Dict[str, str]:
        _get_predictor()
        return {"status": "ready"}

    return app
