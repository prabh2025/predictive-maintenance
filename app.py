import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Load models
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_maintenance(
    machine_id, temperature, vibration,
    pressure, rpm, voltage, current
):
    try:
        # Create dataframe
        data = {
            'temperature': temperature,
            'vibration': vibration,
            'pressure': pressure,
            'rpm': rpm,
            'voltage': voltage,
            'current': current
        }
        df = pd.DataFrame([data])

        # Add engineered features
        feature_cols = [
            'temperature', 'vibration', 'pressure',
            'rpm', 'voltage', 'current'
        ]
        for col in feature_cols:
            df[f'{col}_mean_5'] = df[col]
            df[f'{col}_std_5'] = 0.0
            df[f'{col}_max_5'] = df[col]

        df['temp_vib_ratio'] = (
            df['temperature'] / (df['vibration'] + 0.001)
        )
        df['power'] = df['voltage'] * df['current']

        # Scale and predict
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]

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
- Power: {voltage * current:.1f} W
        """
        return result

    except Exception as e:
        return f"Error: {str(e)}"

# Examples
examples = [
    ["M001", 75.0, 0.5, 100.0, 1500.0, 220.0, 10.0],
    ["M002", 95.0, 0.85, 140.0, 1800.0, 235.0, 14.0],
    ["M003", 82.0, 0.60, 115.0, 1600.0, 225.0, 11.0],
    ["M004", 68.0, 0.42, 95.0, 1400.0, 215.0, 9.0],
]

# Create Gradio UI
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