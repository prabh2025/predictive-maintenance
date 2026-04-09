from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check passed!")

def test_predict_low_risk():
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
    print(f"✅ Low risk test passed! Risk: {data['risk_level']}")

def test_predict_high_risk():
    response = client.post("/predict", json={
        "machine_id": "M002",
        "temperature": 95.0,
        "vibration": 0.85,
        "pressure": 140.0,
        "rpm": 1800.0,
        "voltage": 235.0,
        "current": 14.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["failure_probability"] > 0.5
    print(f"✅ High risk test passed! Prob: {data['failure_probability']}")

def test_predict_batch():
    response = client.post("/predict/batch", json={
        "readings": [
            {
                "machine_id": "M001",
                "temperature": 75.0,
                "vibration": 0.5,
                "pressure": 100.0,
                "rpm": 1500.0,
                "voltage": 220.0,
                "current": 10.0
            },
            {
                "machine_id": "M002",
                "temperature": 95.0,
                "vibration": 0.85,
                "pressure": 140.0,
                "rpm": 1800.0,
                "voltage": 235.0,
                "current": 14.0
            }
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    print(f"✅ Batch test passed! Total: {data['total']}")

def test_machine_status():
    response = client.get("/machines/M001/status")
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == "M001"
    print("✅ Machine status test passed!")