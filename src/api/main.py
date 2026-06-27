"""
Credit Risk Scoring API

Serves the LightGBM + Optuna model trained in credit-risk-ml-pipeline.
Endpoints:
  GET  /health          — liveness + model metadata
  GET  /ready           — readiness (fails until model bundle is loaded)
  POST /predict         — single-applicant default probability
  GET  /drift           — rolling score distribution + drift alert
  GET  /metrics         — Prometheus scrape endpoint
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from src.api import monitoring
from src.api.predictor import get_bundle_meta, load_bundle, predict
from src.api.schemas import (
    DriftReport,
    HealthResponse,
    LoanApplicationRequest,
    PredictionResponse,
)

_start_time = time.time()
_model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_ready
    try:
        load_bundle()
        _model_ready = True
        print("Model bundle loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Predicts the probability that a loan applicant will default. "
        "Built on LightGBM + Optuna, trained on the Home Credit Default Risk dataset. "
        "ROC-AUC 0.754 · KS Statistic 37.9 · Portfolio profit lift £11.1M."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    meta = get_bundle_meta() if _model_ready else {}
    return HealthResponse(
        status="ok" if _model_ready else "degraded",
        model_version=meta.get("model_version", "not_loaded"),
        test_roc_auc=meta.get("test_roc_auc"),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/ready", tags=["Ops"])
def ready():
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model bundle not yet loaded.")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_endpoint(request: Request, application: LoanApplicationRequest):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not ready.")

    t0 = time.perf_counter()
    try:
        result = predict(application)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.perf_counter() - t0
    monitoring.record_prediction(
        prob=result.default_probability,
        grade=result.risk_grade,
        decision=result.decision,
        latency=latency,
    )
    return result


@app.get("/drift", response_model=DriftReport, tags=["Monitoring"])
def drift_report():
    return monitoring.get_drift_report()


@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
def metrics():
    data, content_type = monitoring.prometheus_metrics()
    return Response(content=data, media_type=content_type)
