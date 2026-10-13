FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/heart_disease_model.joblib ./artifacts/heart_disease_model.joblib

EXPOSE 8000

CMD ["uvicorn", "src.heart_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
