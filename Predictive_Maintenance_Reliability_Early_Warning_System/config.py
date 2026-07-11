"""Configuration constants for the PM-REWS application.

Defines all hardcoded values, model paths, defaults, and thresholds
used across the engineering pipeline and the Streamlit dashboard.
"""

import os
from typing import Dict, List, Any

# Project Directory Structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Create directories if they do not exist
for folder in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, ASSETS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Dataset Configurations
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DATA_FILE_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

# --- THIS IS THE FIX ---
# Define MODEL_FILE so training.py and app.py can import it perfectly
MODEL_FILE = os.path.join(MODELS_DIR, "pm_rews_pipeline.joblib")

# Feature Column Groups
RAW_NUMERICAL_FEATURES: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

CAT_FEATURES: List[str] = ["Type"]
TARGET_COL: str = "Machine failure"
DROP_COLS: List[str] = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

# Engineering Coefficients & Physics Constants
KELVIN_TO_CELSIUS_OFFSET: float = 273.15
RAD_PER_SEC_CONVERSION: float = 2 * 3.1415926535 / 60.0

# Model Training Configurations
RANDOM_STATE: int = 42
CV_SPLITS: int = 5
N_ITER_SEARCH: int = 15

# Default Reference Values for Risk Analysis (UCI Dataset Averages)
REF_AIR_TEMP_C: float = 27.0
REF_PROC_TEMP_C: float = 40.0
REF_RPM: float = 1538.0
REF_TORQUE: float = 40.0
REF_TOOL_WEAR: float = 108.0

# UI Theme Configs
UI_TITLE: str = "Dangote Cement Plc - Predictive Maintenance Dashboard"
UI_THEME_COLOR: str = "#0B3C5D"