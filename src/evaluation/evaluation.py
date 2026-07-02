#Evaluating the trained models

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate standard classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Handle cases where roc_auc cannot be computed (e.g. only one class in y_true)
            metrics["roc_auc"] = np.nan
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics

def save_plots(y_true, y_pred, y_prob=None, task_type="classification", output_dir="plots"):
    """
    Generate and save evaluation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if task_type == "classification":
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Plot ROC Curve if probabilities are provided
        if y_prob is not None:
            from sklearn.metrics import roc_curve
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})", color="darkorange", lw=2)
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "roc_curve.png"))
                plt.close()
            except Exception as e:
                print(f"⚠️ Could not plot ROC curve: {e}")
                
    elif task_type == "regression":
        # Actual vs Predicted Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.3, color="teal")
        # Ideal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
        plt.close()
        
        # Residuals Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residuals, alpha=0.3, color="purple")
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals vs Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals_plot.png"))
        plt.close()
