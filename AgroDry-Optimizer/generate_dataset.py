"""
generate_dataset.py
--------------------
Generates a synthetic industrial batch-drying telemetry dataset used to
train and validate the Predictive Process Optimization Engine.

DATA PROVENANCE NOTE (important for reviewers):
Real proprietary telemetry from industrial batch dryers is not publicly
available, so this dataset is a physics-informed synthetic proxy rather
than field-collected sensor data. The output equations below are built
around well-established drying-process relationships:
  - Final moisture % decreases with longer process duration, higher
    inlet temperature, and higher airflow (faster moisture removal).
  - Final moisture % increases with larger batch mass (longer thermal
    penetration time needed) and higher ambient humidity (slower
    evaporation gradient).
  - Specific Energy Consumption (SEC, kWh/ton) rises with inlet
    temperature and process duration relative to batch mass (a classic
    energy-intensity-per-unit-throughput relationship), and is nudged
    upward by high ambient humidity (dryer must work harder).
Gaussian noise is added to each target to emulate sensor and process
variability. Coefficients are directionally realistic (consistent with
published cassava/grain drying-kinetics literature) but are illustrative,
not calibrated against a specific plant. Before production use, they
should be re-fit against real plant telemetry.
"""

import numpy as np
import pandas as pd


def generate_industrial_telemetry(num_samples=1200, random_seed=42):
    """
    Generates a realistic synthetic dataset for batch industrial processing units.
    Includes ambient conditions, operational inputs, and resulting quality/energy outputs.
    """
    np.random.seed(random_seed)

    # 1. Generate Input Features
    batch_ids = [f"BAT-{2026000 + i}" for i in range(1, num_samples + 1)]

    # Batch Mass in Metric Tons
    batch_mass_tons = np.round(np.random.uniform(1.5, 12.0, num_samples), 2)

    # Ambient Temperature (°C) and Relative Humidity (%)
    ambient_temp_c = np.round(np.random.uniform(22.0, 38.0, num_samples), 1)
    ambient_humidity_pct = np.round(np.random.uniform(35.0, 88.0, num_samples), 1)

    # Operational Settings
    inlet_temp_c = np.round(np.random.uniform(60.0, 115.0, num_samples), 1)
    airflow_m3min = np.round(np.random.uniform(120.0, 450.0, num_samples), 1)
    process_duration_min = np.round(np.random.uniform(30.0, 160.0, num_samples), 1)

    # 2. Physics-Informed Output Equations with Controlled Noise
    noise_moisture = np.random.normal(0, 1.2, num_samples)
    noise_energy = np.random.normal(0, 4.5, num_samples)

    # Final Moisture Content (%) Target: ~12%
    # Moisture decreases with duration and inlet temp, increases with batch mass and ambient humidity
    final_moisture_pct = (
        48.0
        - (process_duration_min * 0.22)
        - (inlet_temp_c * 0.14)
        + (batch_mass_tons * 0.75)
        + (ambient_humidity_pct * 0.04)
        - (airflow_m3min * 0.015)
        + noise_moisture
    )
    final_moisture_pct = np.clip(np.round(final_moisture_pct, 2), 6.5, 32.0)

    # Specific Energy Consumption (kWh per Ton)
    # Energy intensity increases with higher inlet temperatures and longer run times per unit mass
    baseline_sec = 210.0
    specific_energy_kwh_ton = (
        (inlet_temp_c * 1.45)
        + ((process_duration_min * 1.1) / (batch_mass_tons * 0.45))
        + (ambient_humidity_pct * 0.2)
        + noise_energy
    )
    specific_energy_kwh_ton = np.clip(np.round(specific_energy_kwh_ton, 2), 110.0, 320.0)

    # Quality Status Classification
    quality_status = []
    for m in final_moisture_pct:
        if 10.0 <= m <= 14.0:
            quality_status.append("In-Spec (Optimal)")
        elif m < 10.0:
            quality_status.append("Over-Dried (High Energy Waste)")
        else:
            quality_status.append("Under-Dried (Spoilage Risk)")

    # 3. Assemble DataFrame
    df = pd.DataFrame({
        "batch_id": batch_ids,
        "batch_mass_tons": batch_mass_tons,
        "ambient_temp_c": ambient_temp_c,
        "ambient_humidity_pct": ambient_humidity_pct,
        "inlet_temp_c": inlet_temp_c,
        "airflow_m3min": airflow_m3min,
        "process_duration_min": process_duration_min,
        "final_moisture_pct": final_moisture_pct,
        "specific_energy_kwh_ton": specific_energy_kwh_ton,
        "quality_status": quality_status
    })

    return df


if __name__ == "__main__":
    df = generate_industrial_telemetry(num_samples=1200)
    df.to_csv("industrial_batch_telemetry.csv", index=False)
    print(" successfully generated 'industrial_batch_telemetry.csv' with 1,200 records.")
    print("\nDataset Summary:")
    print(df.describe())
