import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────
# Train model on startup if not exists
# ─────────────────────────────────────
def train_model_if_needed():
    if os.path.exists("models/best_model.pkl"):
        print("✅ Model already exists — skipping training")
        return

    print("🔄 Training model on startup...")
    os.makedirs("models", exist_ok=True)

    # Generate training data
    np.random.seed(42)
    n = 10000
    temperature = np.random.normal(75, 10, n)
    vibration = np.random.normal(0.5, 0.1, n)
    pressure = np.random.normal(100, 15, n)
    rpm = np.random.normal(1500, 100, n)
    voltage = np.random.normal(220, 5, n)
    current = np.random.normal(10, 1, n)

    # Build feature matrix
    X = np.column_stack([
        temperature, vibration, pressure,
        rpm, voltage, current,
        temperature, np.zeros(n), temperature,
        vibration, np.zeros(n), vibration,
        pressure, np.zeros(n), pressure,
        rpm, np.zeros(n), rpm,
        voltage, np.zeros(n), voltage,
        current, np.zeros(n), current,
        temperature / (vibration + 0.001),
        voltage * current
    ])

    # Labels — 16% failure rate
    y = ((temperature > 88) | (vibration > 0.65)).astype(int)
    print(f"Failure rate: {y.mean():.2%}")

    # Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Save
    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model trained and saved!")

# Train on startup
train_model_if_needed()

# Load models
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ Models loaded successfully!")

# ─────────────────────────────────────
# Prediction function
# ─────────────────────────────────────
def predict_maintenance(
    machine_id, temperature, vibration,
    pressure, rpm, voltage, current
):
    try:
        # Build feature array
        temp_vib_ratio = temperature / (vibration + 0.001)
        power = voltage * current

        X = np.array([[
            temperature, vibration, pressure,
            rpm, voltage, current,
            temperature, 0.0, temperature,
            vibration, 0.0, vibration,
            pressure, 0.0, pressure,
            rpm, 0.0, rpm,
            voltage, 0.0, voltage,
            current, 0.0, current,
            temp_vib_ratio,
            power
        ]])

        # Scale and predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        # Risk level
        if probability >= 0.7:
            risk = "🔴 HIGH RISK"
            recommendation = "⚠️ Immediate maintenance required!"
        elif probability >= 0.4:
            risk = "🟡 MEDIUM RISK"
            recommendation = "📅 Schedule maintenance within 24 hours"
        else:
            risk = "🟢 LOW RISK"
            recommendation = "✅ Machine operating normally"

        result = f"""
## 🏭 Machine: {machine_id}

### Prediction Result:
- **Failure Predicted:** {'Yes ⚠️' if prediction == 1 else 'No ✅'}
- **Failure Probability:** {probability:.1%}
- **Risk Level:** {risk}

### Recommendation:
{recommendation}

### Sensor Readings:
- Temperature: {temperature}°C
- Vibration: {vibration} g
- Pressure: {pressure} PSI
- RPM: {rpm}
- Power: {power:.1f} W
        """
        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ─────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────
examples = [
    ["M001", 75.0, 0.5, 100.0, 1500.0, 220.0, 10.0],
    ["M002", 95.0, 0.85, 140.0, 1800.0, 235.0, 14.0],
    ["M003", 82.0, 0.60, 115.0, 1600.0, 225.0, 11.0],
    ["M004", 68.0, 0.42, 95.0, 1400.0, 215.0, 9.0],
]

demo = gr.Interface(
    fn=predict_maintenance,
    inputs=[
        gr.Textbox(
            label="Machine ID",
            placeholder="e.g. M001",
            value="M001"
        ),
        gr.Slider(
            minimum=50, maximum=120,
            value=75, step=0.5,
            label="Temperature (°C)"
        ),
        gr.Slider(
            minimum=0.1, maximum=1.5,
            value=0.5, step=0.01,
            label="Vibration (g)"
        ),
        gr.Slider(
            minimum=50, maximum=200,
            value=100, step=1,
            label="Pressure (PSI)"
        ),
        gr.Slider(
            minimum=500, maximum=3000,
            value=1500, step=10,
            label="RPM"
        ),
        gr.Slider(
            minimum=180, maximum=260,
            value=220, step=1,
            label="Voltage (V)"
        ),
        gr.Slider(
            minimum=5, maximum=20,
            value=10, step=0.1,
            label="Current (A)"
        ),
    ],
    outputs=gr.Markdown(label="Prediction Result"),
    title="🏭 Predictive Maintenance AI System",
    description="""
## AI-Powered Industrial Equipment Failure Prediction
Predict machine failures **before they happen** using ML.

**Risk Levels:**
- 🟢 LOW — Normal operation
- 🟡 MEDIUM — Monitor closely  
- 🔴 HIGH — Immediate action needed

**Try the examples below or adjust the sliders!**
    """,
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()