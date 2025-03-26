import numpy as np
import streamlit as st
import joblib
import os

st.set_page_config(page_title="Water Quality Analysis", layout="wide")

# Apply custom styles
st.markdown(
    """
    <style>
        .stApp {background-color: #e3f2fd;}
        .sidebar {background-color: #e1bee7; padding: 20px; border-radius: 10px;}
        .css-1v0mbdj {background-color: #ffffff; border-radius: 10px; padding: 20px;}
        .highlight-box {background-color: #90caf9; padding: 10px; border-radius: 10px; font-weight: bold; font-size: 20px;}
        .water-quality-title {font-size: 22px; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True
)

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

# --- WHO & FAO Guidelines for Different Water Uses ---
water_quality_limits = {
    "Drinking": {
        "DO (mg/L)": (6.5, 8), "pH": (6.5, 8.5), "Conductivity (µS/cm)": 500,
        "BOD (mg/L)": 1, "Nitrate (mg/L)": 10, "Turbidity (NTU)": 1,
        "Chloride (mg/L)": 250, "COD (mg/L)": 3, "Ammonia (mg/L)": 0.5, "TDS (mg/L)": 500
    },
    "Domestic": {
        "DO (mg/L)": (5, 8), "pH": (6.0, 9.0), "Conductivity (µS/cm)": 1500,
        "BOD (mg/L)": 5, "Nitrate (mg/L)": 50, "Turbidity (NTU)": 5,
        "Chloride (mg/L)": 600, "COD (mg/L)": 10, "Ammonia (mg/L)": 1, "TDS (mg/L)": 1000
    },
    "Agriculture": {
        "DO (mg/L)": (4, 6), "pH": (6.0, 8.5), "Conductivity (µS/cm)": 3000,
        "BOD (mg/L)": 10, "Nitrate (mg/L)": 50, "Turbidity (NTU)": 10,
        "Chloride (mg/L)": 700, "COD (mg/L)": 20, "Ammonia (mg/L)": 5, "TDS (mg/L)": 2000
    }
}

# --- Streamlit UI ---
st.title("🌊 CCME WQI Calculator & BOD Predictor")

st.sidebar.header("Enter Water Quality Parameters")

# Sidebar inputs
user_inputs = {}
for param in feature_columns:
    user_inputs[param] = st.sidebar.number_input(f"{param}:", min_value=0.0, value=0.0, step=0.1)

# Button to trigger prediction
if st.sidebar.button("🔍 Calculate WQI & Predict BOD"):
    if any(value < 0 for value in user_inputs.values()):
        st.error("All values must be non-negative.")
    else:
        X_input = np.array([user_inputs[col] for col in feature_columns]).reshape(1, -1)
        predicted_bod = rf_model.predict(X_input)[0]

        # Add predicted BOD to the inputs
        user_inputs["BOD (mg/L)"] = predicted_bod

        st.markdown(f"<div class='highlight-box'>Predicted BOD: {predicted_bod:.2f} mg/L</div>", unsafe_allow_html=True)
        
        for category, limits in water_quality_limits.items():
            failed_params = 0
            failed_tests = 0
            deviations = []

            for param, value in user_inputs.items():
                if param in limits:
                    limit = limits[param]
                    if isinstance(limit, tuple):
                        if not (limit[0] <= value <= limit[1]):
                            failed_params += 1
                            failed_tests += 1
                            deviation = min(abs(value - limit[0]), abs(value - limit[1])) / (limit[1] - limit[0]) * 100
                            deviations.append(deviation)
                    else:
                        if value > limit:
                            failed_params += 1
                            failed_tests += 1
                            deviation = ((value / limit) - 1) * 100
                            deviations.append(deviation)

            F1 = (failed_params / len(limits)) * 100  # Scope
            F2 = (failed_tests / len(limits)) * 100  # Frequency
            NSE = np.sum(deviations) / len(limits) if deviations else 0  # Normalized Sum of Excursions
            F3 = NSE / (0.01 * NSE + 0.01)  # Amplitude

            CCME_WQI = 100 - (np.sqrt(F1**2 + F2**2 + F3**2) / 1.732)

            quality = (
                "Excellent" if CCME_WQI >= 95 else
                "Good" if CCME_WQI >= 80 else
                "Fair" if CCME_WQI >= 65 else
                "Marginal" if CCME_WQI >= 45 else
                "Poor"
            )

            with st.expander(f"<span class='water-quality-title'>{category} Water Quality</span>", expanded=False):
                st.info(f"**CCME WQI Score:** {CCME_WQI:.2f}")
                st.write(f"**Water Quality Category:** {quality}")
                st.write("---")
