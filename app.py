import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os

# --- Data and Model Loading ---
@st.cache(allow_output_mutation=True)
def load_model():
    if not os.path.exists("rf_model.pkl") or not os.path.exists("scaler.pkl"):
        st.error("Model files not found. Please run your training script first.")
        st.stop()
    rf_best = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return rf_best, scaler

rf_best, scaler = load_model()

# --- WHO Guidelines for Water Quality ---
limits = {
    "DO (mg/L)": (6.5, 8),
    "pH": (6.5, 8.5),
    "Conductivity (µS/cm)": 400,
    "Nitrate (mg/L)": 10,
    "Turbidity (NTU)": 1,
    "Chloride (mg/L)": 250,
    "COD (mg/L)": 10,
    "Ammonia (mg/L)": 0.5,
    "TDS (mg/L)": 500
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

# sidebar inputs for each parameter (excluding BOD)
user_inputs = {
    param: st.sidebar.number_input(
        label=f"{param}:", min_value=0.0, value=0.0, step=0.1
    )
    for param in limits.keys()
}

# Month selection dropdown
month_choice = st.sidebar.selectbox("Month:", list(month_mapping.keys()))
user_inputs["Month"] = month_mapping[month_choice]

# Button to trigger prediction
if st.sidebar.button("Calculate WQI"):
    # Validate inputs: ensure no negative values
    if any(value < 0 for value in user_inputs.values()):
        st.error("All values must be non-negative.")
    else:
        # Prepare input data for prediction.
        # The model was trained on these features (without BOD).
        X_input = np.array(list(user_inputs.values())).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        predicted_bod = rf_best.predict(X_scaled)[0]
        
        # Create a new dictionary for WQI calculation that includes the predicted BOD.
        wqi_inputs = user_inputs.copy()
        wqi_inputs["BOD (mg/L)"] = predicted_bod
        
        # Calculate CCME WQI components.
        failed_params = 0
        failed_tests = 0
        deviations = []
        for param, value in wqi_inputs.items():
            # Only apply limits to parameters that exist in our limits dictionary
            if param in limits:
                limit = limits[param]
                if isinstance(limit, tuple):
                    if not (limit[0] <= value <= limit[1]):
                        failed_params += 1
                        failed_tests += 1
                        deviations.append(
                            min(abs(value - limit[0]), abs(value - limit[1])) /
                            (limit[1] - limit[0]) * 100
                        )
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
