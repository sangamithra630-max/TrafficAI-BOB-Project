# ============================================================
# IBM BOB PROJECT: TrafficAI - Agentic AI Traffic Optimization
# SDG 11: Sustainable Cities and Communities
# Submitted by: Sangamithra D
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 55)
print("🚦 TrafficAI - IBM BOB Project")
print("=" * 55)

# ---- STEP 1: Create Traffic Data ----
np.random.seed(42)
n = 300

data = pd.DataFrame({
    'hour': np.random.randint(0, 24, n),
    'vehicle_count': np.random.randint(50, 600, n),
    'weather': np.random.randint(1, 4, n),
    'congestion': np.random.randint(1, 10, n)
})

print("\nDataset created with", n, "records")
print(data.head())

# ---- STEP 2: Train ML Model ----
X = data[['hour', 'vehicle_count', 'weather']]
y = data['congestion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "=" * 55)
print("ML MODEL RESULTS")
print("=" * 55)
print("R² Score:", round(r2_score(y_test, y_pred), 3))
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))

# ---- STEP 3: Visualize ----
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Congestion")
plt.ylabel("Predicted Congestion")
plt.title("TrafficAI - BOB Prediction Results")
plt.tight_layout()
plt.savefig("prediction_results.png")
plt.show()

# ---- STEP 4: Agentic AI Layer ----
def monitoring_agent(row):
    return {"vehicle_count": row['vehicle_count'], "weather": row['weather']}

def prediction_agent(data):
    features = np.array([[12, data['vehicle_count'], data['weather']]])
    return model.predict(features)[0]

def routing_agent(pred):
    if pred > 7:
        return "Heavy congestion. Reroute via eco-route B."
    elif pred > 4:
        return "Moderate congestion. Suggest alternate route A."
    else:
        return "Clear roads. Normal route recommended."

def alert_agent(msg):
    print("[ALERT SENT]:", msg)

print("\n" + "=" * 55)
print("AGENTIC AI OUTPUT")
print("=" * 55)

sample = data.iloc[0]
m_data = monitoring_agent(sample)
pred = prediction_agent(m_data)
route = routing_agent(pred)
alert_agent(route)

print("Predicted Congestion:", round(pred, 2))
print("Recommendation:", route)

print("\n" + "=" * 55)
print("✅ BOB PROJECT COMPLETE")
print("=" * 55)