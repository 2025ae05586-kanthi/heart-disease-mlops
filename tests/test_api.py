from fastapi.testclient import TestClient

from src.heart_ml.api import app

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

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post(
        "/predict",
        json=SAMPLE_INPUT,
        )


        assert response.status_code == 200

        result = response.json()

        assert result["prediction"] in [0, 1]
        assert 0.0 <= result["probability_class_0"] <= 1.0
        assert 0.0 <= result["probability_class_1"] <= 1.0

        probability_sum = (
            result["probability_class_0"]
            + result["probability_class_1"]
        )
        assert abs(probability_sum - 1.0) < 1e-6

        assert "not a medical diagnosis" in result["message"]


def test_predict_rejects_invalid_input():
    invalid_input = SAMPLE_INPUT.copy()
    invalid_input["sex"] = 5

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=invalid_input,
        )

        assert response.status_code == 422

