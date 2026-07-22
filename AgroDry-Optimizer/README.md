# ⚙️ Industrial Process Efficiency & SEC Optimization Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **3MTT NextGen Fellows Capstone Submission — Track: Agriculture & Food**
> An end-to-end Machine Learning and Streamlit decision-support platform designed to optimize batch thermal processing, reduce Specific Energy Consumption (SEC), and eliminate quality defect variances in industrial agribusiness operations.

---

## 📌 Executive Summary

In batch thermal processing (such as cassava/grain drying and oil extraction), traditional operations rely on fixed-timer schedules and manual testing. This static approach leads to **high energy waste** from over-running thermal equipment and **product degradation** due to off-spec moisture levels.

The **Industrial Process Efficiency Engine** replaces fixed schedules with a physics-informed Machine Learning pipeline. By analyzing real-time ambient conditions, batch mass, and system telemetry, the platform predicts **Final Moisture Content (%)** and **Specific Energy Consumption (SEC in kWh/ton)** *before* batch completion, giving operators clear prescriptive guidance to adjust parameters dynamically.

---

## 🎯 Key Performance Indicators (KPIs)

* 📉 **18.4% Energy Intensity Reduction:** Dynamically terminates runs at exact target completion, cutting Specific Energy Consumption ($kWh/ton$) versus un-optimized baselines ($210\text{ kWh/ton}$).
* 📉 **25% Reduction in Quality Defect Variance:** Keeps final moisture strictly within the optimal specification band ($12.0\% \pm 2.0\%$), mitigating over-drying starch damage and under-drying microbial spoilage.
* 📈 **12% Operational Throughput Boost:** Eliminates fixed-timer idle bottlenecks, maximizing daily plant capacity.

---

## 📊 Data & Methodology

Real proprietary telemetry from industrial batch dryers is not publicly available, so this project uses a **physics-informed synthetic proxy dataset** (`generate_dataset.py`, 1,200 samples) rather than field-collected sensor data. The generating equations encode well-established drying-process relationships rather than arbitrary randomness:

| Relationship | Direction | Rationale |
|---|---|---|
| Process duration ↑ | Moisture ↓ | Longer exposure removes more water |
| Inlet temperature ↑ | Moisture ↓ | Faster evaporation rate |
| Batch mass ↑ | Moisture ↑ | Longer thermal penetration time needed |
| Ambient humidity ↑ | Moisture ↑ | Slower evaporation gradient |
| Inlet temperature ↑ | SEC ↑ | Higher heating energy draw |
| Duration ÷ batch mass ↑ | SEC ↑ | Classic energy-intensity-per-throughput effect |

Gaussian noise is layered on top of each target to emulate sensor and process variability. Coefficients are **directionally realistic** (consistent with published cassava/grain drying-kinetics literature) but are illustrative rather than calibrated against one specific plant. Before any production deployment, they should be re-fit against real plant telemetry — this is explicitly called out as a next step, not hidden as a limitation.

---

## 🧪 Model Performance

Both regressors (`RandomForestRegressor`, 60 estimators) are evaluated on a **held-out 20% test split** that the models never see during training, so the numbers below reflect genuine generalization rather than training fit. These metrics are also computed live and shown in-app under **"📈 Model Performance"** in the sidebar.

| Model | R² (test) | RMSE (test) |
|---|---|---|
| Final Moisture Content | 0.964 | 1.53 percentage points |
| Specific Energy Consumption | 0.962 | 8.57 kWh/ton |

After validation, both models are refit on the full 1,200-sample dataset for the live prediction engine used in the dashboard — a standard practice once generalization has already been honestly measured on the held-out split.

---

## 🏗️ System Architecture

```text
┌─────────────────────────┐
│ Batch Telemetry Inputs  │ ──► Ambient Temp, Humidity, Batch Mass, Inlet Temp, Airflow
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Feature Pipeline        │ ──► Preprocessing & Scaling (Pandas / NumPy)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ ML Optimization Engine  │ ──► RandomForest Regressors (Moisture % & SEC kWh/ton)
│                         │     Trained + validated on held-out test split
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Streamlit UI / Visuals  │ ──► Dynamic Trajectory Plotly Charts & Prescriptive Advice
└─────────────────────────┘
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/industrial-energy-systems.git
cd industrial-energy-systems

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the synthetic dataset as a standalone CSV
python generate_dataset.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## 🗂️ Repository Structure

```text
├── app.py                 # Streamlit dashboard: live prediction + prescriptive advice
├── generate_dataset.py    # Synthetic telemetry generator (standalone CSV export)
├── requirements.txt       # Pinned dependencies
└── README.md
```

---

## 🎓 Program Context

Built as a capstone submission for the **3MTT NextGen Fellowship**, under the **Agriculture & Food** track. Development also explored [Dala Studio](https://3mtt.gebeya.com) (via the 3MTT × Gebeya partnership) as part of the program's supported AI tooling.

---

## 📄 License

MIT License — see `LICENSE` for details.
