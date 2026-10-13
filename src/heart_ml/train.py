
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "target"

# Coded categorical features should not be treated as continuous measurements.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]


def build_preprocessor():
    """Create separate preprocessing pipelines for numeric and categorical data."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def get_models():
    """Return the candidate models and their fixed configurations."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted model on the held-out test set."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return metrics, predictions, probabilities


def main():
    """Train, cross-validate, compare, and save the best candidate."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run the data preparation step first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    mlflow.set_tracking_uri(
        f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"
    )
    mlflow.set_experiment("heart-disease-classification")

    results = []

    for model_name, estimator in get_models().items():
        print(f"\n{'=' * 60}")
        print(f"Training: {model_name}")
        print(f"{'=' * 60}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

            cv_summary = {
                f"cv_{metric}_mean": float(
                    cv_results[f"test_{metric}"].mean()
                )
                for metric in scoring
            }

            for metric, value in cv_summary.items():
                mlflow.log_metric(metric, value)

            pipeline.fit(X_train, y_train)

            metrics, predictions, probabilities = evaluate_model(
                pipeline,
                X_test,
                y_test,
            )

            mlflow.log_params(
                {
                    "model": model_name,
                    "test_size": 0.20,
                    "random_state": 42,
                    "cv_folds": 5,
                    "categorical_features": ",".join(
                        CATEGORICAL_FEATURES
                    ),
                    "numerical_features": ",".join(
                        NUMERICAL_FEATURES
                    ),
                }
            )

            mlflow.log_metrics(
                {
                    f"test_{metric}": float(value)
                    for metric, value in metrics.items()
                }
            )

            print("\nCross-validation means:")
            for metric, value in cv_summary.items():
                print(f"{metric}: {value:.4f}")

            print("\nHeld-out test metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\nClassification report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            )

            print("Confusion matrix:")
            print(confusion_matrix(y_test, predictions))

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
            )

            results.append(
                {
                    "model_name": model_name,
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "cv_summary": cv_summary,
                    "confusion_matrix": confusion_matrix(
                        y_test, predictions
                    ).tolist(),
                }
            )

    # Choose the model using training-only cross-validation ROC-AUC.
    best_result = max(
        results,
        key=lambda result: result["cv_summary"]["cv_roc_auc_mean"],
    )

    best_model = best_result["pipeline"]
    model_path = ARTIFACTS_DIR / "heart_disease_model.joblib"

    joblib.dump(best_model, model_path)

    summary = {
        "selection_metric": "cv_roc_auc_mean",
        "selected_model": best_result["model_name"],
        "models": [
            {
                "model_name": result["model_name"],
                "cv_metrics": result["cv_summary"],
                "test_metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"],
            }
            for result in results
        ],
    }

    summary_path = REPORTS_DIR / "model_results.json"
    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print(f"\nSelected model: {best_result['model_name']}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {summary_path}")
    print("\nStart the MLflow UI with:")
    print(
        r".\.venv\Scripts\python.exe -m mlflow ui "
        r"--backend-store-uri .\mlruns"
    )


if __name__ == "__main__":
    main()