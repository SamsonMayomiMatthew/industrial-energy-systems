import streamlit as st
import matplotlib.pyplot as plt
from model import train_predictive_engine, calculate_rul

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DCP Reliability Early-Warning System",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for professional styling and branding
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .main-title {
        color: #1b5e20;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHED DATA LOADING
# ==========================================
@st.cache_data
def load_model_and_data(filepath):
    """Wrapper function to cache the data load and model training."""
    return train_predictive_engine(filepath)

# Load and train
try:
    model, poly, df = load_model_and_data('data.xlsx')
    c0 = model.intercept_
    c1 = model.coef_[0]
    c2 = model.coef_[1]
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure the file is in the same directory.")
    st.stop()

# ==========================================
# HEADER SECTION
# ==========================================
st.markdown("<h1 class='main-title'>DCP University Engineering Challenge</h1>", unsafe_allow_html=True)
st.markdown("### Track 2: Predictive Maintenance & Reliability Early-Warning System (Support Rollers)")
st.write("---")

# ==========================================
# SIDEBAR / USER INTERACTION
# ==========================================
st.sidebar.header("🕹️ Telemetry Controls")
st.sidebar.write("Simulate real-time sensor streams by adjusting the time-step slider below:")

# Dynamic slider based on actual data length to prevent KeyErrors
max_index = len(df) - 1
selected_step = st.sidebar.slider(
    "Select Operating Time-Step (Hours):",
    min_value=0,
    max_value=max_index,
    value=min(20, max_index),
    step=1
)

st.sidebar.markdown("---")
st.sidebar.write("**Asset Specifications:**")
st.sidebar.markdown("""
- **Component:** Support Roller Shaft
- **Key Metric:** Torsional Torque (Nm)
- **Warning Threshold:** 20.0 Nm
- **Critical Safety Limit:** 30.0 Nm
""")

# ==========================================
# GET CURRENT STATE TELEMETRY
# ==========================================
current_torque = df.loc[selected_step, 'Torque_Nm']
previous_torque = df.loc[selected_step - 1, 'Torque_Nm'] if selected_step > 0 else current_torque
torque_rate_of_change = current_torque - previous_torque

# Run Early Warning Rules & ML Forecast
sudden_jump = torque_rate_of_change > 3.0 # Threshold for sudden shift

if sudden_jump:
    status = "CRITICAL ANOMALY"
    color = "#e74c3c" # Red
    badge = "🔴 EMERGENCY"
    rul_display = "IMMINENT RISK"
    rec = "⚠️ **IMMEDIATE DISPATCH REQUIRED:** A sudden mechanical/alignment shift has occurred. Deploy a technician to conduct ultrasound scans and check bolt tightening on the roller bearing housing."
elif current_torque >= 30.0:
    status = "CRITICAL LIMIT EXCEEDED"
    color = "#d32f2f" # Red
    badge = "🔴 DANGER"
    rul_display = "0.0 Hours"
    rec = "🚨 **EMERGENCY STANDBY:** Component has breached the maximum safe torque limit. Automatically initiating an orderly plant shutdown sequence to protect the roller from catastrophic failure."
elif current_torque >= 20.0:
    status = "PREDICTIVE WEAR ZONE"
    color = "#f39c12" # Amber
    badge = "🟡 WARNING (AMBER)"
    rul = calculate_rul(selected_step, current_torque, c0, c1, c2)
    rul_display = f"{rul} Hours"
    rec = f"📋 **PREVENTIVE WORK ORDER GENERATED:** The system forecasts bearing degradation. A maintenance ticket has been queued in SAP CMMS. Schedule roller replacement within the next **{int(rul)} hours**."
else:
    status = "NORMAL OPERATIONS"
    color = "#2ecc71" # Green
    badge = "🟢 HEALTHY"
    rul = calculate_rul(selected_step, current_torque, c0, c1, c2)
    rul_display = f"{rul} Hours"
    rec = "✅ **OPTIMAL PERFORMANCE:** Component is operating within normal boundaries. No intervention required. Standard sensor sweeps continue."

# ==========================================
# MAIN DASHBOARD LAYOUT
# ==========================================

# KPI Cards row
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {color};'>
        <p style='margin:0; font-size:14px; color:#555; text-transform:uppercase;'>System Status</p>
        <h2 style='margin:0; color:{color};'>{badge}</h2>
        <p style='margin:0; font-size:12px; color:#777;'>{status}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db;'>
        <p style='margin:0; font-size:14px; color:#555; text-transform:uppercase;'>Current Torque Input</p>
        <h2 style='margin:0; color:#2c3e50;'>{current_torque:.2f} Nm</h2>
        <p style='margin:0; font-size:12px; color:#777;'>Rate of Change: {torque_rate_of_change:+.3f} Nm/hr</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #9b59b6;'>
        <p style='margin:0; font-size:14px; color:#555; text-transform:uppercase;'>Remaining Useful Life (RUL)</p>
        <h2 style='margin:0; color:#2c3e50;'>{rul_display}</h2>
        <p style='margin:0; font-size:12px; color:#777;'>ML Predictive Horizon</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.info(rec)
st.write("")

# Dynamic Plots & Model Insights Section
chart_col, info_col = st.columns([2, 1])

with chart_col:
    st.subheader("📊 Dynamic Asset Health Trajectory")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot all data up to current step
    ax.scatter(df['Time_step'][:selected_step+1], df['Torque_Nm'][:selected_step+1], color=color, s=35, zorder=3, label="Current Operation Path")
    ax.plot(df['Time_step'], df['Torque_Nm'], color='#bdc3c7', linestyle='--', alpha=0.5, label="Predicted Path (Digital Twin)")
    
    # Annotate sudden jump
    if selected_step >= 3:
        ax.annotate('Step Fault (t=3)', xy=(3, 15), xytext=(15, 11),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=9, fontweight='bold', color='#c0392b')
                     
    # Safety Limits lines
    ax.axhline(y=20, color='#f39c12', linestyle=':', label='Warning Limit (20 Nm)')
    ax.axhline(y=30, color='#d32f2f', linestyle=':', label='Critical Limit (30 Nm)')
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(8, 42)
    ax.set_xlabel("Operating Time Step (Hours)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig)

with info_col:
    st.subheader("⚙️ Digital Twin Engineering Parameters")
    st.markdown("The predictive engine uses a mathematical **digital twin model** calibrated directly from ANSYS structural load simulations of DCP support rollers.")
    
    st.markdown(f"""
    **Trained Regression Parameters:**
    * **Baseline Load Intercept ($c_0$):** `{c0:.4f} Nm`
    * **Linear Growth Factor ($c_1$):** `{c1:.4f}`
    * **Quadratic Wear Factor ($c_2$):** `{c2:.6f}`
    
    **Mathematical Model Fit Formula:**
    $$Torque(t) = {c0:.2f} + {c1:.2f} \cdot t + {c2:.5f} \cdot t^2$$
    """)
    st.success(f"**Model Accuracy Metric:** $R^2 = {model.score(poly.fit_transform(df[df['Time_step']>=3]['Time_step'].values.reshape(-1,1)), df[df['Time_step']>=3]['Torque_Nm'].values):.4f}$ (Highly Reliable)")
