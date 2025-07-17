import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump  # ⬅️ Replaces pickle

# Simulate fake driver data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'average_speed': np.random.normal(60, 10, n),
    'brake_events': np.random.poisson(2, n),
    'acceleration_events': np.random.exponential(1, n),
    'night_trip': np.random.choice([0, 1], n),
    'trip_duration_min': np.random.normal(30, 10, n),
    'distance_km': np.random.normal(15, 5, n),
})

# Simulate a risk score based on rules
data['risk_score'] = (
    (data['average_speed'] > 80).astype(int) +
    (data['brake_events'] > 3).astype(int) +
    (data['acceleration_events'] > 2).astype(int) +
    data['night_trip'] +
    (data['trip_duration_min'] > 60).astype(int)
)

# Create risk_category: 0=Low, 1=Medium, 2=High
conditions = [
    (data['risk_score'] <= 1),
    (data['risk_score'] == 2),
    (data['risk_score'] >= 3)
]
choices = [0, 1, 2]
data['risk_category'] = np.select(conditions, choices)

X = data.drop(['risk_score', 'risk_category'], axis=1)
y = data['risk_category']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model using joblib
dump(model, 'model.pkl')

print("✅ Model trained and saved to model.pkl (with joblib)")
