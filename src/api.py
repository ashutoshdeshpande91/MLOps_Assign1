"""FastAPI service for heart disease risk prediction.

Endpoints:
    GET  /health    liveness/readiness check (also tells you if the model loaded)
    POST /predict   JSON patient record in, prediction + confidence out
    GET  /metrics   Prometheus metrics (request counts, latency, predictions)

Run locally (from the repo root):
    uvicorn api:app --app-dir src --reload   # then open http://127.0.0.1:8000/docs
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from predict import FEATURES, SAMPLE_PATIENT, load_model, predict

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/heart_model.joblib"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

# --- Prometheus metrics (scraped from GET /metrics) ---
REQUESTS = Counter(
    "api_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
LATENCY = Histogram(
    "api_request_duration_seconds", "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter(
    "api_predictions_total", "Predictions made, by predicted class", ["label"]
)
CONFIDENCE = Histogram(
    "api_prediction_confidence", "Confidence of the predicted class",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)


class PatientRecord(BaseModel):
    """One patient's health measurements (UCI Cleveland heart disease features)."""

    model_config = ConfigDict(json_schema_extra={"example": SAMPLE_PATIENT})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels coloured by fluoroscopy")
    thal: int = Field(..., ge=3, le=7, description="3 normal, 6 fixed defect, 7 reversible defect")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not")
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(..., description="Probability of heart disease")
    label: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except FileNotFoundError as err:
        logger.error("%s", err)
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with a request id, status code and latency."""
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_id=%s %s %s failed", request_id,
                         request.method, request.url.path)
        raise
    elapsed = time.perf_counter() - start
    ms = elapsed * 1000
    logger.info("request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
                request_id, request.method, request.url.path,
                response.status_code, ms)
    if request.url.path != "/metrics":
        # use the route template (not the raw URL) so labels stay few
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True, "features": FEATURES}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(record: PatientRecord, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predict(model, record.model_dump())[0]
    result["label"] = "heart disease" if result["prediction"] == 1 else "no heart disease"
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    logger.info("prediction=%s confidence=%.3f", result["prediction"], result["confidence"])
    return result


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
