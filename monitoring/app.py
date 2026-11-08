import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from starlette.responses import Response

MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "attrition_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Employee Attrition API", version="1.0.0")

# ---- Prometheus metrics ----
REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions made, by predicted class", ["predicted_class"])
INPUT_INCOME = Histogram("model_input_monthly_income", "Monthly income values received by /predict",
                         buckets=[20000, 40000, 60000, 80000, 100000, 120000, 150000])
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded")
MODEL_LOADED.set(1)


class Employee(BaseModel):
    age: int = Field(ge=18, le=70)
    monthly_income: int = Field(ge=0)
    years_at_company: int = Field(ge=0, le=50)
    overtime: int = Field(ge=0, le=1)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    LATENCY.labels(request.url.path).observe(elapsed)
    REQUESTS.labels(request.url.path, request.method, str(response.status_code)).inc()
    return response


@app.get("/")
def root():
    return {"message": "Employee Attrition API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(data: Employee):
    df = pd.DataFrame([data.model_dump()])
    pred = int(model.predict(df)[0])
    PREDICTIONS.labels(str(pred)).inc()
    INPUT_INCOME.observe(data.monthly_income)
    return {"prediction": pred, "will_leave": bool(pred)}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
