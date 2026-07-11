# 🏭 Predictive Maintenance & Reliability Early-Warning System (PM-REWS)
### *An Industrial-Grade Machine Learning Pipeline for Dangote Cement Plc*

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Executive Overview & Business Value

In heavy-industry manufacturing like cement production, unplanned asset downtime is a catastrophic vector for capital degradation. If a primary milling or kiln system experiences a sudden structural breakdown, the entire facility halts—causing severe hourly production losses and compounding maintenance costs.

**PM-REWS** converts volatile, raw machinery telemetry into structured, highly accurate, and proactive operational field directives. By mapping physical indicators through specialized feature engineering, this platform calculates real-time failure probabilities and automates risk mitigation workflows before critical physical assets reach a breakpoint.

### 🎯 Key Engineering Innovations:
*   **Physics-Driven Feature Engineering:** Transforms raw telemetry into deep indicators mapping mechanical power, thermal degradation, and cumulative stress envelopes.
*   **Leakproof Resampling Pipelines:** Utilizes localized Stratified Cross-Validation bound with SMOTE to handle extreme class imbalances without overoptimistic data leakage.
*   **Adaptive Executive Dashboard:** A reactive, no-click Streamlit deployment built for non-technical plant managers to track asset health states instantly.

---

## 📊 Asset Criticality Framework

To bridge the gap between abstract dataset tags (`L`, `M`, `H`) and standard factory operations, the system relies on an industry-standard **Asset Criticality Classification** matrix:

| Dataset Tag | Asset Criticality Class | Operational Reality | Cement Plant Mapping |
| :---: | :--- | :--- | :--- |
| **L** | **Auxiliary Support Equipment** | Non-essential equipment. Failure does not impact raw cement throughput; repairs can be scheduled normally. | Packing bag printers, local exhaust fans |
| **M** | **Operational Line Assets** | Line-critical machinery. Failure halts a specific line, but the facility can buffer outputs or bypass for a few hours. | Additive weigh feeders, clinker cooling fans |
| **H** | **Plant-Critical Asset** | **Single Point of Failure.** Failure completely stops factory production, creating immediate financial losses. | Main Limestone Crusher, Raw Mill, Rotary Kiln |

---

## 🗂️ Project Repository Structure

```text
PM_REWS/
├── .gitignore                   # Safely bypasses massive data weights and cache environments
├── README.md                    # System architecture and technical documentation
├── requirements.txt             # Locked-down production dependencies
├── config.py                    # Global path constants and threshold baselines
├── feature_engineering.py       # Custom physics-based metric transformation functions
├── model_selection.py          # Streamlined ensembling exploration script
├── maintenance_engine.py       # Rule-based threshold escalation engine
├── app.py                       # Reactive Streamlit monitoring UI dashboard
├── data/
│   └── data_placeholder.txt     # Local cache path for raw AI4I 2020 source datasets
├── models/
│   └── .gitkeep                 # Directory containing trained serialized serialization artifacts
└── assets/
    └── README.md                # Placeholders for branding styles and UI metrics
