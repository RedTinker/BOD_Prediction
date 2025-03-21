import numpy as np
import streamlit as st
import joblib
import os

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists("rf_model.pkl"):
        st.error("Model file not found. Please run your training script first.")
        st.stop()
    return joblib.load("rf_model.pkl")

rf_model = load_model()

# --- Load Feature Names ---
@st.cache_resource
def load_feature_names():
    if not os.path.exists("feature_names.pkl"):
        st.error("Feature names file not found. Ensure feature_names.pkl is available.")
        st.stop()
    return joblib.load("feature_names.pkl")

feature_columns = load_feature_names()

# --- WHO Guidelines for Water Quality ---
limits = {
    "DO (mg/L)": (6.5, 8), "pH": (6.5, 8.5), "Conductivity (µS/cm)": 400,
    "Nitrate (mg/L)": 10, "Turbidity (NTU)": 1, "Chloride (mg/L)": 250,
    "COD (mg/L)": 10, "Ammonia (mg/L)": 0.5, "TDS (mg/L)": 500
}

# --- Month Mapping ---
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# --- Streamlit UI ---
st.title("CCME WQI Calculator & BOD Predictor")

st.sidebar.header("Enter Water Quality Parameters")

# Sidebar inputs
user_inputs = {}
for param in feature_columns:
    if param == "Month":
        user_inputs[param] = month_mapping[st.sidebar.selectbox("Month:", list(month_mapping.keys()))]
    else:
        user_inputs[param] = st.sidebar.number_input(f"{param}:", min_value=0.0, value=0.0, step=0.1)

# Button to trigger prediction
if st.sidebar.button("Calculate WQI"):
    if any(value < 0 for value in user_inputs.values()):
        st.error("All values must be non-negative.")
    else:
        X_input = np.array([user_inputs[col] for col in feature_columns]).reshape(1, -1)
        predicted_bod = rf_model.predict(X_input)[0]

        # Add predicted BOD to the inputs
        wqi_inputs = user_inputs.copy()
        wqi_inputs["BOD (mg/L)"] = predicted_bod

        # Calculate CCME WQI components
        failed_params = 0
        failed_tests = 0
        deviations = []
        for param, value in wqi_inputs.items():
            if param in limits:
                limit = limits[param]
                if isinstance(limit, tuple):
                    if not (limit[0] <= value <= limit[1]):
                        failed_params += 1
                        failed_tests += 1
                        deviations.append(min(abs(value - limit[0]), abs(value - limit[1])) / (limit[1] - limit[0]) * 100)
                else:
                    if value > limit:
                        failed_params += 1
                        failed_tests += 1
                        deviations.append(((value / limit) - 1) * 100)

        F1 = (failed_params / len(limits)) * 100
        F2 = (failed_tests / len(limits)) * 100
        NSE = np.sum(deviations) / len(limits) if deviations else 0
        F3 = NSE / (0.01 * NSE + 0.01)
        CCME_WQI = 100 - (np.sqrt(F1**2 + F2**2 + F3**2) / 1.732)

        quality = (
            "Excellent" if CCME_WQI >= 95 else
            "Good" if CCME_WQI >= 80 else
            "Fair" if CCME_WQI >= 65 else
            "Marginal" if CCME_WQI >= 45 else
            "Poor"
        )

        st.success(f"Predicted BOD: {predicted_bod:.2f} mg/L")
        st.info(f"CCME WQI Score: {CCME_WQI:.2f}")
        st.write(f"Water Quality Category: {quality}")
