"""Model Training and Artifact Serialization Module."""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import clean_and_prepare_dataset, create_preprocessing_pipeline
from model_selection import get_candidate_models, optimize_hyperparameters
from evaluation import optimize_threshold
from config import MODEL_FILE, RANDOM_STATE
from utils import LOGGER

def run_training_pipeline(df: pd.DataFrame):
    """Executes the complete end-to-end model training and artifact generation."""
    X, y = clean_and_prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    preprocessor = create_preprocessing_pipeline()
    candidates = get_candidate_models()
    
    best_overall_model = None
    best_overall_score = 0.0
    
    for name, config in candidates.items():
        LOGGER.info(f"Optimizing: {name}")
        search = optimize_hyperparameters(name, config, preprocessor, X_train, y_train)
        
        if search.best_score_ > best_overall_score:
            best_overall_score = search.best_score_
            best_overall_model = search.best_estimator_
            
    # Optimize threshold
    y_probs = best_overall_model.predict_proba(X_test)[:, 1]
    threshold, f1 = optimize_threshold(y_test, y_probs)
    
    # Save artifacts
    artifacts = {
        "pipeline": best_overall_model,
        "threshold": threshold,
        "f1": f1
    }
    joblib.dump(artifacts, MODEL_FILE)
    LOGGER.info(f"Model saved. Optimal Threshold: {threshold:.4f} (F1: {f1:.4f})")