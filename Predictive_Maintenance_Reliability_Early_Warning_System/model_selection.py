"""Model Selection and Hyperparameter Optimization Pipeline.

Defines the search configurations, performs class imbalance adjustments via SMOTE
inside imblearn pipelines, and runs cross-validated hyperparameter optimization.
"""

from typing import Dict, Any
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

from config import RANDOM_STATE, CV_SPLITS, N_ITER_SEARCH

def get_candidate_models() -> Dict[str, Dict[str, Any]]:
    """Returns candidate estimators matched with tailored search hyperparameter grids."""
    
    grids = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__class_weight": ["balanced", None]
            }
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [8, 12, 20],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 4, 5]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7],
                "classifier__scale_pos_weight": [1, 5, 10]
            }
        }
    }
    return grids

def optimize_hyperparameters(
    model_name: str, 
    model_config: Dict[str, Any], 
    preprocessor: Any, 
    X_train: Any, 
    y_train: Any
) -> RandomizedSearchCV:
    """Executes randomized grid optimization wrapped around a leakproof resampling pipeline.

    Args:
        model_name: String label identifier.
        model_config: Inner configuration structure with estimator and search grids.
        preprocessor: Configured feature ColumnTransformer pipeline.
        X_train: Training features matrix.
        y_train: Training target array.

    Returns:
        Optimized RandomizedSearchCV execution container.
    """
    # Build complete end-to-end pipeline with embedded SMOTE balancing
    full_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", model_config["model"])
    ])
    
    cv_strategy = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=model_config["param_grid"],
        n_iter=N_ITER_SEARCH,
        scoring="f1",
        cv=cv_strategy,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    return search