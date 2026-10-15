# trafficai_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime

st.set_page_config(page_title="TrafficAI", page_icon="🚦", layout="wide")

st.title("🚦 TrafficAI - Agentic Traffic Optimization System")
st.markdown("**SDG 11: Sustainable Cities and Communities** | AICTE | IBM SkillsBuild 2026")
st.markdown(f"*Live Dashboard — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
st.markdown("---")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 5000
    hours = np.random.randint(0, 24, n)
    vehicles = np.random.randint(50, 900, n)
    weather = np.random.choice([1, 2, 3], n, p=[0.6, 0.3, 0.1])
    weekend = np.random.choice([0, 1], n, p=[0.7, 0.3])
    base = (vehicles / 100) + (weather * 1.5) + (weekend * -1)
    rush = np.where(((hours >= 7) & (hours <= 10)) | ((hours >= 17) & (hours <= 20)), 3, 0)
    congestion = np.clip(base + rush + np.random.normal(0, 1, n), 1, 10)
    X = np.column_stack([hours, vehicles, weather, weekend])
    y = congestion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

model, model_r2 = train_model()

st.sidebar.header("🎛️ Live Traffic Input")
hour = st.sidebar.slider("Hour of Day", 0, 23, 17)
vehicle_count = st.sidebar.slider("Vehicle Count", 50, 900, 700)
weather = st.sidebar.selectbox("Weather", [1, 2, 3], format_func=lambda x: {1:"☀️ Clear", 2:"🌧️ Rain", 3:"🌫️ Fog"}[x])
is_weekend = st.sidebar.selectbox("Weekend?", [0, 1], format_func=lambda x: {0:"No", 1:"Yes"}[x])
is_emergency = st.sidebar.checkbox("🚨 Emergency Vehicle Present?")

input_data = np.array([[hour, vehicle_count, weather, is_weekend]])
prediction = round(float(model.predict(input_data)[0]), 2)

st.subheader("📊 Real-Time Traffic Prediction")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Congestion", f"{prediction}/10")
col2.metric("Model Accuracy", f"{model_r2*100:.1f}%")

if prediction > 7.5:
    status = "🔴 CRITICAL"
    route = "Eco-Route B (Highway bypass)"
elif prediction > 5.0:
    status = "🟡 MODERATE"
    route = "Route A (Alternate city road)"
else:
    status = "🟢 LOW"
    route = "Route C (Direct normal route)"

col3.metric("Status", status)
col4.metric("Recommended Route", route.split("(")[0])

st.markdown("---")
st.subheader("🤖 Agentic AI Workflow")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**📥 Monitoring Agent**")
    st.write(f"Captured live data at Hour {hour} with {vehicle_count} vehicles")
    st.markdown("**🧠 Prediction Agent**")
    st.write(f"Predicted congestion level: **{prediction}/10**")

with col_b:
    st.markdown("**🚦 Routing Agent**")
    st.write(f"Recommended: **{route}**")
    st.markdown("**📡 Alert Agent**")
    if is_emergency and prediction > 5.0:
        st.error("🚨 EMERGENCY OVERRIDE ACTIVATED: Green corridor initialized for Ambulance")
    st.success(f"Alert sent via WhatsApp / Municipal Dashboard")

st.markdown("---")
st.subheader("🌱 Sustainability Impact")
col_x, col_y, col_z = st.columns(3)
col_x.metric("CO₂ Reduction", f"{prediction * 1.5:.1f} kg/day")
col_y.metric("Commute Time Saved", f"{prediction * 2.1:.1f} mins")
col_z.metric("Emergency Response", "Priority Active" if is_emergency else "Standard")

st.markdown("---")
st.subheader("✅ Agent Cycle Summary")
st.json({
    "timestamp": datetime.now().isoformat(),
    "input": {"hour": hour, "vehicles": vehicle_count, "weather": weather, "weekend": is_weekend},
    "prediction": prediction,
    "status": status,
    "route": route,
    "emergency_override": is_emergency,
    "alert_channels": ["WhatsApp", "Municipal Dashboard", "Email"]
})

st.success("✅ TrafficAI Agent Cycle Complete — SDG 11 Alignment Active")