from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(
    title="Predictive Maintenance API",
    description="AI-powered industrial equipment failure prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Input model
class SensorData(BaseModel):
    machine_id: str
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    voltage: float
    current: float

class BatchSensorData(BaseModel):
    readings: List[SensorData]

# Response model
class PredictionResponse(BaseModel):
    machine_id: str
    failure_predicted: bool
    failure_probability: float
    risk_level: str
    recommendation: str
    latency_ms: float

# Helper function
def get_risk_level(probability):
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendation(risk_level):
    recommendations = {
        "HIGH": "Immediate maintenance required!",
        "MEDIUM": "Schedule maintenance within 24 hours",
        "LOW": "Machine operating normally"
    }
    return recommendations[risk_level]

def predict_failure(data: dict):
    df = pd.DataFrame([data])
    df = df.drop(['machine_id'], axis=1)

    # Add engineered features manually
    feature_cols = [
        'temperature', 'vibration', 'pressure',
        'rpm', 'voltage', 'current'
    ]

    # Rolling features (use same value since single reading)
    for col in feature_cols:
        df[f'{col}_mean_5'] = df[col]
        df[f'{col}_std_5'] = 0.0
        df[f'{col}_max_5'] = df[col]

    # Interaction features
    df['temp_vib_ratio'] = (
        df['temperature'] / (df['vibration'] + 0.001)
    )
    df['power'] = df['voltage'] * df['current']

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return int(prediction), float(probability)
# Endpoints
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "predictive_maintenance",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(sensor_data: SensorData):
    try:
        start = time.time()
        data = sensor_data.dict()

        prediction, probability = predict_failure(data)
        risk = get_risk_level(probability)
        recommendation = get_recommendation(risk)

        latency = (time.time() - start) * 1000

        return PredictionResponse(
            machine_id=data['machine_id'],
            failure_predicted=bool(prediction),
            failure_probability=round(probability, 4),
            risk_level=risk,
            recommendation=recommendation,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(batch: BatchSensorData):
    try:
        results = []
        for reading in batch.readings:
            data = reading.dict()
            prediction, probability = predict_failure(data)
            risk = get_risk_level(probability)
            results.append({
                "machine_id": data['machine_id'],
                "failure_predicted": bool(prediction),
                "failure_probability": round(probability, 4),
                "risk_level": risk,
                "recommendation": get_recommendation(risk)
            })
        return {
            "predictions": results,
            "total": len(results),
            "high_risk": sum(
                1 for r in results
                if r['risk_level'] == 'HIGH'
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/machines/{machine_id}/status")
def machine_status(machine_id: str):
    return {
        "machine_id": machine_id,
        "status": "operational",
        "last_checked": "2026-04-07T10:00:00"
    }