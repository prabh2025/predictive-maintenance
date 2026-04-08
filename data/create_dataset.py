import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

# Generate sensor data
temperature = np.random.normal(75, 10, n)
vibration = np.random.normal(0.5, 0.1, n)
pressure = np.random.normal(100, 15, n)
rpm = np.random.normal(1500, 100, n)
voltage = np.random.normal(220, 5, n)
current = np.random.normal(10, 1, n)

# Create dataframe
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
    'machine_id': np.random.choice(['M001','M002','M003','M004','M005'], n),
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'rpm': rpm,
    'voltage': voltage,
    'current': current,
})

# Create failure — 12% rate
# High temp OR high vibration causes failure
data['failure'] = 0
data.loc[temperature > 88, 'failure'] = 1
data.loc[vibration > 0.65, 'failure'] = 1

# Save
data.to_csv('data/raw/sensor_data.csv', index=False)
print(f"Dataset created: {data.shape}")
print(f"Failure rate: {data['failure'].mean():.2%}")
print(f"Total failures: {data['failure'].sum()}")