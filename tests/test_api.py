from fastapi.testclient
import TestClient
import sys
sys.path.append('.')
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "machine_id": "M001",
        "temperature": 75.0,
        "vibration": 0.5,
        "pressure": 100.0,
        "rpm": 1500.0,
        "voltage": 220.0,
        "current": 10.0
    })
    assert response.status_code == 200
    data = response.json()
    assert "failure_predicted" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_high_risk_prediction():
    response = client.post("/predict", json={
        "machine_id": "M002",
        "temperature": 98.0,
        "vibration": 0.85,
        "pressure": 140.0,
        "rpm": 1800.0,
        "voltage": 235.0,
        "current": 14.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["failure_probability"] > 0.5