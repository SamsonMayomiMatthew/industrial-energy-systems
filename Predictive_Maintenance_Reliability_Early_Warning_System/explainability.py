"""Explainability Module incorporating SHAP (SHapley Additive exPlanations).

Provides diagnostic model audit details showing raw telemetry weight contributions
to plant failures.
"""

import os
from typing import Any, List
import matplotlib.pyplot as plt
import pandas as pd
import shap
from config import OUTPUTS_DIR

def run_shap_analysis(pipeline: Any, X_sample: pd.DataFrame) -> shap.Explainer:
    """Extracts global feature weights using a SHAP Explainer pipeline.

    Args:
        pipeline: The fully trained model/preprocessing execution bundle.
        X_sample: Evaluation feature set matrix.

    Returns:
        Configured SHAP Explainer model.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # Preprocess values for evaluation compatibility
    X_trans = preprocessor.transform(X_sample)
    
    # Get feature names from transformer architecture mapping
    try:
        feat_names = list(preprocessor.get_feature_names_out())
    except:
        feat_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
        
    X_trans_df = pd.DataFrame(X_trans, columns=feat_names)
    
    # Handle model-specific explainers
    if hasattr(classifier, "tree_method") or "GradientBoosting" in str(type(classifier)):
        explainer = shap.TreeExplainer(classifier)
    else:
        explainer = shap.Explainer(classifier, X_trans_df)
        
    shap_values = explainer(X_trans_df, check_additivity=False)
    
    # Export static summary plot to local filesystem
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "shap_summary.png"), dpi=150)
    plt.close()
    
    return explainer