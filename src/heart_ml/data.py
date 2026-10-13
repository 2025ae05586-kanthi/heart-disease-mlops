
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/processed.cleveland.data"
)

COLUMNS = [
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
    "num",
]

FEATURE_COLUMNS = COLUMNS[:-1]
TARGET_COLUMN = "target"


def download_data(force=False):
    """Download the original Cleveland dataset if it is unavailable."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    path = RAW_DIR / "processed.cleveland.data"

    if force or not path.exists():
        print(f"Downloading dataset from UCI: {DATA_URL}")
        urlretrieve(DATA_URL, path)

    return path


def load_raw_data(path=None):
    """Load the dataset and interpret '?' as missing values."""
    if path is None:
        path = download_data()

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )

    for column in COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def clean_data(df):
    """Create a binary target while retaining missing feature values."""
    missing_columns = set(COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df[COLUMNS].copy()

    # Remove rows where the original target is missing.
    result = result.dropna(subset=["num"])

    # Convert the original target into a binary classification label.
    result[TARGET_COLUMN] = (
        result["num"] > 0
    ).astype("int64")

    # Remove the original target column.
    result = result.drop(columns=["num"])

    if not result[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if result.empty:
        raise ValueError("The cleaned dataset is empty.")

    return result


def prepare_dataset():
    """Download, clean, save, and summarize the dataset."""
    raw = load_raw_data()
    cleaned = clean_data(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "heart_disease.csv"
    cleaned.to_csv(output_path, index=False)

    print("\n--- Dataset Summary ---")
    print(f"Raw shape: {raw.shape}")
    print(f"Cleaned shape: {cleaned.shape}")

    print("\nMissing values by column:")
    print(cleaned.isna().sum())

    print("\nTarget class counts:")
    print(cleaned["target"].value_counts().sort_index())

    print("\nTarget proportions:")
    print(
        cleaned["target"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print(f"\nSaved cleaned dataset to: {output_path}")

    return cleaned


if __name__ == "__main__":
    prepare_dataset()