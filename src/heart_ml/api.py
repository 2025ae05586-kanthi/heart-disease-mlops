
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.heart_ml.predict import load_model, predict_one


model = None

MONITORING_LOG_FILE = Path(
    os.getenv("MONITORING_LOG_FILE", "monitoring/requests.jsonl")
)
MONITORING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
monitoring_lock = Lock()


def write_request_log(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Append one request record to the shared JSONL log."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "success": 200 <= status_code < 400,
    }

    with monitoring_lock:
        with MONITORING_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record) + "\n")


class HeartDiseaseInput(BaseModel):
    """Validate the 13 features expected by the model."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    probability_class_0: float
    probability_class_1: float
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the API starts."""
    global model

    model = load_model()
    yield
    model = None


app = FastAPI(
    title="Heart Disease Classification API",
    description=(
        "Educational machine-learning demonstration. "
        "Predictions are not medical diagnoses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Measure requests and persist their outcomes."""
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        write_request_log(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
        )


@app.get("/health")
def health():
    """Return service and model readiness."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HeartDiseaseInput):
    """Predict the model's binary target for one feature record."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        result = predict_one(
            model,
            input_data.model_dump(),
        )
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        **result,
        "message": (
            "Educational model output only; "
            "not a medical diagnosis."
        ),
    }
