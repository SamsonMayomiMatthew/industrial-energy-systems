import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def train_predictive_engine(filepath):
    """Loads dataset and trains the polynomial regression model."""
    # Using openpyxl ensures cloud servers unzip modern excel archives correctly
    df = pd.read_excel(filepath, engine="openpyxl")
    
    # Train only on the active degradation phase (Time_step >= 3)
    df_train = df[df['Time_step'] >= 3]
    
    X = df_train['Time_step'].values.reshape(-1, 1)
    y = df_train['Torque_Nm'].values
    
    # Fit quadratic polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly, df

def calculate_rul(current_time, current_torque, c0, c1, c2, critical_limit=30.0):
    """Calculates the Remaining Useful Life (RUL) using the quadratic formula."""
    if current_torque >= critical_limit:
        return 0.0
    
    a = c2
    b = c1
    c = c0 - critical_limit
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return np.nan
    
    t_fail = (-b + np.sqrt(discriminant)) / (2*a)
    rul = max(0.0, t_fail - current_time)
    
    return round(rul, 1)