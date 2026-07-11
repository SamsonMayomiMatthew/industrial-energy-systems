"""Preprocessing Module for Data Pipelines.

Constructs modular Scikit-Learn transformers and pipelines to robustly clean,
impute, and encode production cement facility telemetry.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import RAW_NUMERICAL_FEATURES, CAT_FEATURES, DROP_COLS, TARGET_COL
from feature_engineering import engineer_features

def get_feature_lists() -> Tuple[List[str], List[str]]:
    """Determines dynamic feature schemas after engineering expansion.

    Returns:
        Tuple containing lists of numerical and categorical feature names.
    """
    # Sample a dummy layout to extract names programmatically
    dummy_df = pd.DataFrame(columns=RAW_NUMERICAL_FEATURES + CAT_FEATURES)
    engineered_dummy = engineer_features(dummy_df)
    
    engineered_numerical = [
        col for col in engineered_dummy.columns 
        if col not in CAT_FEATURES and col != TARGET_COL
    ]
    return engineered_numerical, CAT_FEATURES

def create_preprocessing_pipeline() -> ColumnTransformer:
    """Creates a production-grade ColumnTransformer for data scaling and encoding.

    Returns:
        Configured ColumnTransformer pipeline.
    """
    num_cols, cat_cols = get_feature_lists()
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def clean_and_prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Cleans, engineers features, and partitions inputs into features and target.

    Args:
        df: Raw DataFrame input.

    Returns:
        Tuple containing feature DataFrame X and target Series y.
    """
    df_cleaned = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df_engineered = engineer_features(df_cleaned)
    
    X = df_engineered.drop(columns=[TARGET_COL], errors="ignore")
    y = df_engineered[TARGET_COL] if TARGET_COL in df_engineered.columns else None
    return X, y