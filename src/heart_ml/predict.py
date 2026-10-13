
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "heart_disease_model.joblib"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_model():
    """Load the trained preprocessing and classification pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the training script first."
        )

    return joblib.load(MODEL_PATH)


def predict_one(model, features):
    """Return a binary prediction and its estimated class probability."""
    missing = set(FEATURE_COLUMNS) - set(features.keys())
    extra = set(features.keys()) - set(FEATURE_COLUMNS)

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected features: {sorted(extra)}"
        )

    row = pd.DataFrame(
        [{column: features[column] for column in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return {
        "prediction": prediction,
        "probability_class_0": float(probabilities[0]),
        "probability_class_1": float(probabilities[1]),
    }