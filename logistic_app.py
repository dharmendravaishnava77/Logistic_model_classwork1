import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("│ Delivery Delay Prediction App")
st.write("Enter the delivery details below to predict if the shipment will be delayed or on-time.")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('logistic.sav')

model = load_model()

# Define the layout using columns
col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
    traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
    weather_condition = st.slider("Weather Condition Score (1-5)", min_value=1, max_value=5, value=3)
    delivery_slot = st.slider("Delivery Slot (Hour of Day/Shift)", min_value=1, max_value=4, value=2)
    driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=50, value=5, step=1)
    num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=3, step=1)

with col2:
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5, step=1)
    road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
    package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=1.0, max_value=50.0, value=15.0, step=0.1)
    warehouse_processing_time = st.number_input("Warehouse Processing Time (minutes)", min_value=0, max_value=300, value=45, step=1)

# Predict button
if st.button("Predict Delay Status", type="primary"):
    # Construct the input feature array matching the model structure
    features = np.array([[
        delivery_distance, 
        traffic_congestion, 
        weather_condition, 
        delivery_slot, 
        driver_experience, 
        num_stops, 
        vehicle_age, 
        road_condition_score, 
        package_weight, 
        fuel_efficiency, 
        warehouse_processing_time
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    st.markdown("--- ")
    if prediction == 1:
        st.error(f"☑ Prediction: **Delayed** (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.success(f"✅ Prediction: **On Time** (Confidence: {prediction_proba[0]*100:.2f}%)")
