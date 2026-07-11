"""Feature Engineering Module for calculating specialized engineering indices."""

import numpy as np
import pandas as pd
from config import KELVIN_TO_CELSIUS_OFFSET, RAD_PER_SEC_CONVERSION

def add_thermal_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates temperature difference between tool and room ambient environment."""
    df["Thermal_Delta"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df

def add_mechanical_power_w(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates mechanical power generated in Watts based on Torque and Angular Speed."""
    angular_speed = df["Rotational speed [rpm]"] * RAD_PER_SEC_CONVERSION
    df["Mechanical_Power_W"] = df["Torque [Nm]"] * angular_speed
    return df

def add_accumulated_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Computes operational stress based on rotational friction load and tool runtime."""
    df["Accumulated_Stress"] = df["Torque [Nm]"] * df["Tool wear [min]"]
    return df

def add_temperature_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Computes ratio between operating process environment and ambient air."""
    df["Temperature_Ratio"] = df["Process temperature [K]"] / df["Air temperature [K]"]
    return df

def add_power_to_wear(df: pd.DataFrame) -> pd.DataFrame:
    """Computes power usage vs current degradation level safely to prevent inf/NaN."""
    power = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * RAD_PER_SEC_CONVERSION)
    df["Power_to_Wear"] = power / (df["Tool wear [min]"] + 1.0)
    return df

def add_torque_per_rpm(df: pd.DataFrame) -> pd.DataFrame:
    """Computes localized mechanical force distribution ratio per unit of speed."""
    df["Torque_per_RPM"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1.0)
    return df

def add_thermal_stress_index(df: pd.DataFrame) -> pd.DataFrame:
    """Combines heat gradient with tool aging timeline to show thermal exhaustion risk."""
    delta = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Thermal_Stress_Index"] = delta * df["Tool wear [min]"]
    return df

def add_load_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates deviation of torque from common running baselines."""
    df["Load_Factor"] = df["Torque [Nm]"] / (df["Torque [Nm]"].median() + 1e-5)
    return df

def add_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates an empirical inverse degradation index from stress indicators."""
    norm_torque = df["Torque [Nm]"] / 80.0
    norm_wear = df["Tool wear [min]"] / 250.0
    df["Health_Index"] = 100.0 - ((norm_torque * 0.5 + norm_wear * 0.5) * 100.0)
    df["Health_Index"] = df["Health_Index"].clip(0.0, 100.0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all the domain-specific industrial feature engineering functions.

    Args:
        df: Input raw data DataFrame.

    Returns:
        DataFrame augmented with engineered signals.
    """
    df = df.copy()
    df = add_thermal_delta(df)
    df = add_mechanical_power_w(df)
    df = add_accumulated_stress(df)
    df = add_temperature_ratio(df)
    df = add_power_to_wear(df)
    df = add_torque_per_rpm(df)
    df = add_thermal_stress_index(df)
    df = add_load_factor(df)
    df = add_health_index(df)
    return df