"""FraudGuard REST API (FastAPI).

Endpoints
    GET  /health         liveness + model/database status
    GET  /model-info     model type, features, threshold, validation/test metrics
    POST /predict        score one transaction (JSON) with SHAP explanation
    POST /predict-batch  score a CSV upload (multipart) - returns per-row results
    GET  /history        recent predictions from SQLite
    GET  /history/stats  aggregate counts

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from api.schemas import (
    BatchRowResult,
    BatchSummary,
    ErrorResponse,
    HealthResponse,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from fraudguard import __version__
from fraudguard.config import INPUT_COLUMNS
from fraudguard.db import PredictionStore, get_store
from fraudguard.predictor import FraudPredictor, InputValidationError, get_predictor

logger = logging.getLogger("fraudguard.api")

MAX_BATCH_ROWS = 50_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = None
    app.state.store = None
    try:
        app.state.predictor = get_predictor()
    except Exception as exc:  # pragma: no cover - surfaced via /health
        logger.exception("Model failed to load: %s", exc)
    try:
        app.state.store = get_store()
    except Exception as exc:  # pragma: no cover
        logger.exception("Database failed to initialise: %s", exc)
    yield


app = FastAPI(
    title="FraudGuard API",
    description="Explainable transaction fraud detection (PaySim, XGBoost, SHAP).",
    version=__version__,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)


def _predictor(request: Request) -> FraudPredictor:
    p = request.app.state.predictor
    if p is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train it with `python -m fraudguard.train`.",
        )
    return p


def _store(request: Request) -> PredictionStore | None:
    return request.app.state.store


# --------------------------------------------------------------------------- #
# Structured error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422, content={"error": "validation_error", "detail": exc.errors()}
    )


@app.exception_handler(InputValidationError)
async def input_error_handler(_: Request, exc: InputValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code, content={"error": "http_error", "detail": exc.detail}
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred"},
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root():
    """Send visitors of the bare host to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request):
    p = request.app.state.predictor
    store = request.app.state.store
    db_ok = False
    if store is not None:
        try:
            store.stats()
            db_ok = True
        except Exception:  # pragma: no cover
            db_ok = False
    return HealthResponse(
        status="ok" if (p is not None and db_ok) else "degraded",
        model_loaded=p is not None,
        model_type=p.model_type if p else None,
        version=__version__,
        database="ok" if db_ok else "unavailable",
    )


@app.get("/model-info", tags=["system"])
def model_info(request: Request):
    return _predictor(request).info()


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(body: PredictRequest, request: Request):
    predictor = _predictor(request)
    result = predictor.predict_one(
        body.transaction.model_dump(),
        threshold=body.threshold,
        explain=body.explain,
        top_n=body.top_n,
    )
    payload = result.to_dict()

    prediction_id = None
    store = _store(request)
    if store is not None:
        scored = pd.DataFrame(
            [
                {
                    **body.transaction.model_dump(),
                    **payload,
                    "reasons": payload.get("explanation", {}).get("reasons", []),
                }
            ]
        )
        try:
            store.save_frame(
                scored, source="api", model_version=predictor.metadata.get("trained_at_utc")
            )
            prediction_id = store.recent(limit=1)[0]["id"]
        except Exception as exc:  # pragma: no cover - history must never break scoring
            logger.warning("Could not persist prediction: %s", exc)

    return PredictResponse(**payload, model_type=predictor.model_type, prediction_id=prediction_id)


@app.post("/predict-batch", response_model=PredictBatchResponse, tags=["scoring"])
async def predict_batch(
    request: Request,
    file: UploadFile = File(..., description="CSV with columns " + ", ".join(INPUT_COLUMNS)),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    explain: bool = Query(True, description="Attach reason codes to each row"),
):
    predictor = _predictor(request)
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload must be a .csv file")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc
    if len(df) == 0:
        raise HTTPException(status_code=422, detail="CSV contains no rows")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413, detail=f"CSV has {len(df)} rows; maximum is {MAX_BATCH_ROWS}"
        )

    t0 = time.perf_counter()
    scored = predictor.predict_frame(df, threshold=threshold, explain=explain, top_n=3)
    used_threshold = predictor.threshold if threshold is None else threshold
    scored["threshold"] = used_threshold

    batch_id = "unsaved"
    store = _store(request)
    if store is not None:
        try:
            batch_id = store.save_frame(
                scored, source="api-batch", model_version=predictor.metadata.get("trained_at_utc")
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not persist batch: %s", exc)

    results = [
        BatchRowResult(
            row=int(i),
            fraud_probability=float(r.fraud_probability),
            is_fraud=bool(r.is_fraud),
            risk_level=str(r.risk_level),
            reasons=list(r.reasons) if explain and "reasons" in scored.columns else [],
        )
        for i, r in enumerate(scored.itertuples(index=False))
    ]
    flagged = int(scored["is_fraud"].sum())
    return PredictBatchResponse(
        summary=BatchSummary(
            rows=len(scored),
            flagged=flagged,
            flag_rate=flagged / len(scored),
            threshold=float(used_threshold),
            batch_id=batch_id,
            processing_ms=round((time.perf_counter() - t0) * 1000, 1),
        ),
        results=results,
    )


@app.get("/history", tags=["history"])
def history(request: Request, limit: int = Query(50, ge=1, le=1000), only_fraud: bool = False):
    store = _store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Prediction history database unavailable")
    return {"items": store.recent(limit=limit, only_fraud=only_fraud)}


@app.get("/history/stats", tags=["history"])
def history_stats(request: Request):
    store = _store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="Prediction history database unavailable")
    return store.stats()
