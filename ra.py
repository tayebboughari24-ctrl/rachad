import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# Page Configuration
st.set_page_config(
    page_title="Natural Disaster Prediction",
    page_icon="🌍",
    layout="wide"
)

# Header Section
st.title("🌍 Natural Disaster Prediction System")
st.markdown("""
    This application analyzes historical environmental factors to predict the likelihood 
    of **Floods** and **Seismic Activities** using machine learning logic.
""")

# Sidebar Navigation
st.sidebar.header("Navigation")
category = st.sidebar.radio("Select Analysis Type:", ["Flood Prediction", "Seismic Activity"])

# --- SECTION 1: FLOOD PREDICTION ---
if category == "Flood Prediction":
    st.header("🌊 Flood Likelihood Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.slider("Annual Rainfall (mm)", 500, 5000, 1200)
        river_level = st.slider("Current River Level (m)", 0.0, 15.0, 3.5)
    
    with col2:
        soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 45)
        urbanization = st.slider("Urbanization/Deforestation (%)", 0, 100, 20)

    # Simplified prediction logic (Substitute with a trained ML model)
    risk_score = (rainfall * 0.4 + river_level * 30 + soil_moisture * 0.2 + urbanization * 0.1) / 100
    
    if st.button("Calculate Flood Risk"):
        st.subheader("Result:")
        if risk_score > 7.5:
            st.error(f"High Alert: Significant Flood Risk Detected! (Index: {risk_score:.2f})")
        elif risk_score > 4.5:
            st.warning(f"Moderate Risk: Monitor local water levels closely. (Index: {risk_score:.2f})")
        else:
            st.success(f"Low Risk: Current conditions are stable. (Index: {risk_score:.2f})")

# --- SECTION 2: SEISMIC ACTIVITY ---
elif category == "Seismic Activity":
    st.header("🌋 Seismic Activity & Earthquake Risk")
    
    col1, col2 = st.columns(2)
    with col1:
        hist_magnitude = st.number_input("Historical Max Magnitude (Richter Scale)", 0.0, 9.5, 5.0)
        fault_dist = st.slider("Distance from Nearest Fault Line (km)", 0, 500, 50)
    
    with col2:
        depth = st.slider("Expected Focal Depth (km)", 0, 700, 35)
        tectonic_plate = st.selectbox("Tectonic Activity Level", ["Low", "Moderate", "High", "Critical"])

    if st.button("Assess Seismic Risk"):
        st.info("Assessment Summary:")
        st.write(f"Based on a historical magnitude of {hist_magnitude} and a distance of {fault_dist}km from the fault line, the area is categorized under **{tectonic_plate}** activity monitoring.")
        
        # Placeholder for Map Visualization
        st.subheader("Regional Vulnerability Map")
        map_data = pd.DataFrame(
            np.random.randn(5, 2) / [50, 50] + [37.76, -122.4], # Sample coordinates
            columns=['lat', 'lon']
        )
        st.map(map_data)

# --- DATASET SECTION ---
st.divider()
st.subheader("📊 Historical Data Upload")
st.info("Upload your historical environmental datasets (CSV format) to retrain the model or view trends.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())
