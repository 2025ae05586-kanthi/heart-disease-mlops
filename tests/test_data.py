
import pandas as pd
import pytest

from src.heart_ml.data import clean_data


def test_target_is_converted_to_binary():
    df = pd.DataFrame(
        {
            "age": [50, 60, 70],
            "sex": [1, 0, 1],
            "cp": [1, 2, 3],
            "trestbps": [120, 130, 140],
            "chol": [200, 220, 240],
            "fbs": [0, 0, 1],
            "restecg": [0, 1, 0],
            "thalach": [150, 140, 130],
            "exang": [0, 1, 0],
            "oldpeak": [0.0, 1.0, 2.0],
            "slope": [1, 2, 2],
            "ca": [0, 1, 2],
            "thal": [3, 6, 7],
            "num": [0, 1, 4],
        }
    )

    cleaned = clean_data(df)

    assert cleaned["target"].tolist() == [0, 1, 1]
    assert "num" not in cleaned.columns


def test_missing_feature_values_are_preserved():
    df = pd.DataFrame(
        {
            "age": [50, 60],
            "sex": [1, 0],
            "cp": [1, 2],
            "trestbps": [120, 130],
            "chol": [200, 220],
            "fbs": [0, 0],
            "restecg": [0, 1],
            "thalach": [150, 140],
            "exang": [0, 1],
            "oldpeak": [0.0, 1.0],
            "slope": [1, 2],
            "ca": [0, float("nan")],
            "thal": [3, 6],
            "num": [0, 1],
        }
    )

    cleaned = clean_data(df)

    assert len(cleaned) == 2
    assert cleaned["ca"].isna().sum() == 1


def test_missing_required_column_is_rejected():
    df = pd.DataFrame(
        {
            "age": [50],
            "num": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        clean_data(df)