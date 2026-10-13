
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.heart_ml.predict import load_model, predict_one


model = None


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