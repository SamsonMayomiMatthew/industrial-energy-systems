# app.py (Streamlit Front-End Interface)
import streamlit as st
import pandas as pd
import joblib
from config import MODEL_FILE
from charts import get_risk_gauge
from maintenance_engine import generate_maintenance_plan
from feature_engineering import engineer_features

st.set_page_config(page_title="DCP PM-REWS", layout="wide")

# Custom CSS to inject clean modern styling into metric blocks
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    div.stButton > button:first-child { background-color: #0B3C5D; color: white; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("🏭 Cement Plant Reliability Center")
st.markdown("### Real-Time Predictive Maintenance & Early Warning Dashboard")
st.markdown("---")

# 1. Load trained pipeline model artifacts
artifacts = joblib.load(MODEL_FILE)
model = artifacts['pipeline']
threshold = artifacts['threshold']

# 2. Sidebar Live Inputs
st.sidebar.header("Live Asset Monitoring Stream")
# Business-focused mapping that maps back to the raw 'L', 'M', 'H' dataset strings
business_mapping = {
    "🟢 Auxiliary Support Equipment (Low Risk / L)": "L",
    "🟡 Operational Line Assets (Medium Risk / M)": "M",
    "🔴 Plant-Critical Asset (High Revenue Risk / H)": "H"
}

# Display business-friendly titles to the user
selected_display_name = st.sidebar.selectbox("Asset Criticality Level", options=list(business_mapping.keys()), index=2)

# Pass the hidden 'L', 'M', or 'H' string back to your backend data pipeline
asset_type = business_mapping[selected_display_name]
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1000, 2500, 1500)
torque = st.sidebar.slider("Torque [Nm]", 10, 80, 40)
wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 100)

# 3. Dynamic Calculation: Code evaluates instantly as sliders move!
raw_input_data = pd.DataFrame({
    'Type': [asset_type],
    'Air temperature [K]': [300], 
    'Process temperature [K]': [310], 
    'Rotational speed [rpm]': [rpm], 
    'Torque [Nm]': [torque], 
    'Tool wear [min]': [wear]
})

# Complete feature transformation on the fly
input_data = engineer_features(raw_input_data)
prob = model.predict_proba(input_data)[0, 1]
plan = generate_maintenance_plan(prob, 0.95)

# 4. Layout Generation
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📊 Asset Health Score")
    st.plotly_chart(get_risk_gauge(plan['Risk_Score']), use_container_width=True)

with col2:
    st.subheader("📋 Operational Directive")
    
    # Adaptive status alert box based on risk severity
    if plan['Priority'] == 'P1':
        st.error(f"🚨 CRITICAL ALERT: {plan['Recommended_Action']}")
        delta_label = "Critical Anomaly"
    elif plan['Priority'] == 'P2':
        st.warning(f"⚠️ ACTION REQUIRED: {plan['Recommended_Action']}")
        delta_label = "Warning State"
    else:
        st.success(f"✅ SYSTEM STABLE: {plan['Recommended_Action']}")
        delta_label = "Optimal Baseline"
        
    st.markdown("---")
    
    # Multi-column metrics display
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Equipment Risk Index", 
            value=f"{plan['Risk_Score']}%", 
            delta=delta_label, 
            delta_color="inverse" if plan['Risk_Score'] > 35 else "normal"
        )
    with m2:
        st.metric(label="Escalation Priority", value=plan['Priority'])
    with m3:
        st.metric(label="Confidence Level", value=plan['Confidence'])
        
    st.markdown("---")
    
    # Maintenance context layout rows
    st.markdown(f"**🔧 Assigned Engineering Team:** `{plan['Team']}`")
    st.markdown(f"**⏳ Target Remediation Window:** `{plan['Est_Downtime']}`")
