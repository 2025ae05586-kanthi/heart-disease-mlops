
from src.heart_ml.predict import load_model, predict_one


SAMPLE_INPUT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def test_saved_model_can_be_loaded():
    model = load_model()

    assert model is not None


def test_prediction_has_expected_structure():
    model = load_model()
    result = predict_one(model, SAMPLE_INPUT)

    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["probability_class_0"] <= 1.0
    assert 0.0 <= result["probability_class_1"] <= 1.0

    probability_sum = (
        result["probability_class_0"]
        + result["probability_class_1"]
    )

    assert abs(probability_sum - 1.0) < 1e-6


def test_missing_features_are_rejected():
    model = load_model()
    incomplete_input = SAMPLE_INPUT.copy()
    incomplete_input.pop("age")

    try:
        predict_one(model, incomplete_input)
    except ValueError as error:
        assert "Missing features" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for missing features"
        )