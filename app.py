import numpy as np
import streamlit as st
import joblib
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json

st.set_page_config(page_title="Water Quality Analysis", layout="wide")

# Firebase init
@st.cache_resource
def init_firebase():
    key_dict = json.loads(st.secrets["firebase_key"])
    cred = credentials.Certificate(key_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()

@st.cache_resource
def load_model():
    if not os.path.exists("rf_model.pkl"):
        st.error("Model file not found. Please run your training script first.")
        st.stop()
    return joblib.load("rf_model.pkl")

rf_model = load_model()

@st.cache_resource
def load_feature_names():
    if not os.path.exists("feature_names.pkl"):
        st.error("Feature names file not found. Ensure feature_names.pkl is available.")
        st.stop()
    return joblib.load("feature_names.pkl")

feature_columns = load_feature_names()

water_quality_limits = {
    "Potable": {
        "DO (mg/L)": (6.0, 8.0), "pH": (6.5, 8.5), "Conductivity (µS/cm)": 500,
        "BOD (mg/L)": 2, "Nitrate (mg/L)": 45, "Turbidity (NTU)": 1,
        "Chloride (mg/L)": 250, "COD (mg/L)": 3, "Ammonia (mg/L)": 0.5, "TDS (mg/L)": 500
    },
    "Domestic": {
        "DO (mg/L)": (5, 8), "pH": (6.0, 8.5), "Conductivity (µS/cm)": 1500,
        "BOD (mg/L)": 3, "Nitrate (mg/L)": 50, "Turbidity (NTU)": 5,
        "Chloride (mg/L)": 600, "COD (mg/L)": 10, "Ammonia (mg/L)": 1, "TDS (mg/L)": 1000
    },
    "Agriculture": {
        "DO (mg/L)": (4, 8), "pH": (6.5, 8.5), "Conductivity (µS/cm)": 2500,
        "BOD (mg/L)": 10, "Nitrate (mg/L)": 50, "Turbidity (NTU)": 10,
        "Chloride (mg/L)": 700, "COD (mg/L)": 50, "Ammonia (mg/L)": 5, "TDS (mg/L)": 2000
    }
}

def calculate_ccme_wqi(inputs, limits):
    failed_variables = 0
    failed_tests = 0
    excursions = []
    total_variables = len(limits)

    for param, limit in limits.items():
        value = inputs.get(param, np.nan)
        if pd.isna(value):
            continue

        if isinstance(limit, tuple):
            if not (limit[0] <= value <= limit[1]):
                failed_variables += 1
                failed_tests += 1
                deviation = min(abs(value - limit[0]), abs(value - limit[1])) / (limit[1] - limit[0])
                excursions.append(deviation)
        else:
            if value > limit:
                failed_variables += 1
                failed_tests += 1
                excursion = (value / limit) - 1
                excursions.append(excursion)

    F1 = (failed_variables / total_variables) * 100 if total_variables else 0
    F2 = (failed_tests / total_variables) * 100 if total_variables else 0
    NSE = np.sum(excursions) / total_variables if excursions else 0
    F3 = NSE / (0.01 * NSE + 0.01) if NSE else 0

    CCME_WQI = 100 - (np.sqrt(F1**2 + F2**2 + F3**2) / 1.732)

    if CCME_WQI >= 95:
        return CCME_WQI, "Excellent"
    elif CCME_WQI >= 80:
        return CCME_WQI, "Good"
    elif CCME_WQI >= 65:
        return CCME_WQI, "Fair"
    elif CCME_WQI >= 45:
        return CCME_WQI, "Marginal"
    else:
        return CCME_WQI, "Poor"

st.title("🌊 CCME WQI Calculator & BOD Predictor")

st.sidebar.header("Enter Water Quality Parameters")
user_inputs = {}
month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

for param in feature_columns:
    if param == "Month":
        month_index = st.sidebar.slider(f"{param}:", min_value=1, max_value=12, value=1, step=1, format="%d")
        user_inputs[param] = month_index
        st.sidebar.write(f"Selected Month: {month_names[month_index - 1]}")
    else:
        user_inputs[param] = st.sidebar.number_input(f"{param}:", min_value=0.0, value=0.0, step=0.1)

if st.sidebar.button("🔍 Calculate WQI & Predict BOD"):
    if any(value < 0 for key, value in user_inputs.items() if key != "Month"):
        st.error("All values must be non-negative.")
    else:
        X_input = np.array([user_inputs[col] for col in feature_columns]).reshape(1, -1)
        predicted_bod = rf_model.predict(X_input)[0]
        user_inputs["BOD (mg/L)"] = predicted_bod

        st.markdown(
            f"<div style='background-color:#90caf9;padding:10px;border-radius:10px;font-weight:bold;'>"
            f"Predicted BOD: {predicted_bod:.2f} mg/L</div>",
            unsafe_allow_html=True
        )
        
        filtered_inputs = {key: value for key, value in user_inputs.items() if key != "Month"}
        results = {}
        
        for category, limits in water_quality_limits.items():
            CCME_WQI, quality = calculate_ccme_wqi(filtered_inputs, limits)
            results[category] = {"CCME_WQI": CCME_WQI, "Category": quality}
            with st.expander(f"{category} Water Quality", expanded=False):
                st.info(f"**CCME WQI Score:** {CCME_WQI:.2f}")
                st.write(f"**Water Quality Category:** {quality}")

        db.collection("predictions").add({
            "inputs": user_inputs,
            "predicted_bod": predicted_bod,
            "wqi_results": results,
            "timestamp": datetime.utcnow()
        })
