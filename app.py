import numpy as np
import streamlit as st
import joblib
import os

# Set page title and layout
st.set_page_config(page_title="Water Quality Analysis", layout="wide")

# --- Custom Styling ---
st.markdown(
    """
    <style>
        .stApp {background-color: #f4f7fc;}
        .stSidebar {background-color: #ffffff; padding: 20px;}
        .main-title {color: #003366; text-align: center; font-size: 2.2em;}
        .stButton>button {width: 100%; background-color: #0066cc; color: white; border-radius: 10px;}
        .metric-box {border-radius: 10px; background-color: #ffffff; padding: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
        .info-box {border-radius: 10px; background-color: #e3f2fd; padding: 15px;}
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

# --- Water Quality Limits ---
water_quality_limits = {
    "Drinking": {"DO (mg/L)": (6.5, 8), "pH": (6.5, 8.5), "Conductivity (µS/cm)": 500, "BOD (mg/L)": 1},
    "Domestic": {"DO (mg/L)": (5, 8), "pH": (6.0, 9.0), "Conductivity (µS/cm)": 1500, "BOD (mg/L)": 5},
    "Agriculture": {"DO (mg/L)": (4, 6), "pH": (6.0, 8.5), "Conductivity (µS/cm)": 3000, "BOD (mg/L)": 10}
}

# --- Streamlit UI ---
st.markdown("<h1 class='main-title'>🌊 Water Quality & BOD Prediction</h1>", unsafe_allow_html=True)

st.sidebar.header("Enter Water Quality Parameters")

# Sidebar inputs
user_inputs = {}
for param in feature_columns:
    user_inputs[param] = st.sidebar.number_input(f"{param}:", min_value=0.0, value=0.0, step=0.1)

if st.sidebar.button("🔍 Calculate WQI & Predict BOD"):
    if any(value < 0 for value in user_inputs.values()):
        st.error("All values must be non-negative.")
    else:
        X_input = np.array([user_inputs[col] for col in feature_columns]).reshape(1, -1)
        predicted_bod = rf_model.predict(X_input)[0]
        user_inputs["BOD (mg/L)"] = predicted_bod

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='metric-box'><h3>Predicted BOD</h3>", unsafe_allow_html=True)
            st.success(f"{predicted_bod:.2f} mg/L")
            st.markdown("</div>", unsafe_allow_html=True)

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

            with col2:
                st.markdown(f"<div class='info-box'><h3>{category} Water Quality</h3>", unsafe_allow_html=True)
                st.info(f"**CCME WQI Score:** {CCME_WQI:.2f}")
                st.write(f"**Water Quality Category:** {quality}")
                st.markdown("</div>", unsafe_allow_html=True)
