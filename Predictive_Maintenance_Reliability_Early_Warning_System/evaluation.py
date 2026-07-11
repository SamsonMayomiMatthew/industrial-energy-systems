"""Evaluation module generating operational analytics performance profiles.

Optimizes probability operating decision thresholds via Precision-Recall curves
and exports diagnostic graphics to physical memory files.
"""

import os
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_curve, auc, 
    confusion_matrix, classification_report
)
from config import OUTPUTS_DIR

def optimize_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """Locates the optimal probability cutoff threshold using Precision-Recall profiles.

    Args:
        y_true: Ground truth binaries.
        y_probs: Predicted failure probabilities.

    Returns:
        Tuple containing (optimal threshold value, highest achievable F1 score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 for each threshold option safely
    f1_scores = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        preds = (y_probs >= t).astype(int)
        f1_scores[i] = f1_score(y_true, preds, zero_division=0)
        
    if len(f1_scores) == 0:
        return 0.5, 0.0
        
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

def export_evaluation_plots(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> None:
    """Generates and exports physical PNG metrics plots for executive review.

    Args:
        y_true: Validation ground truth.
        y_probs: Risk probabilities.
        threshold: Decision threshold.
    """
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # 2. Confusion Matrix
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center", size="xx-large")
    plt.xlabel("Predictions", fontsize=12)
    plt.ylabel("Actuals", fontsize=12)
    plt.title("Confusion Matrix Dashboard", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()