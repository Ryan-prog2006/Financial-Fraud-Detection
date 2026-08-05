"""Evaluation metrics and plotting utilities for FinShield classical ML models."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve
)
import mlflow

def get_predictions(model, X_test: pd.DataFrame, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Obtains predictions and scoring from a model.

    Args:
        model: Trained model object.
        X_test: Test features.
        model_name: The name of the model.

    Returns:
        A tuple of (y_pred, y_scores).
    """
    if model_name == "isolation_forest":
        preds = model.predict(X_test)
        y_pred = np.where(preds == -1, 1, 0)
        y_scores = -model.decision_function(X_test)
    else:
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
    return y_pred, y_scores

def compute_metrics(y_test: pd.Series, y_pred: np.ndarray, y_scores: np.ndarray, model_name: str) -> dict:
    """Computes standard metrics and prints them in a formatted table.

    Args:
        y_test: Test targets.
        y_pred: Predicted target labels.
        y_scores: Predicted target scores.
        model_name: The model name.

    Returns:
        A dictionary containing the calculated metrics.
    """
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nMetrics for {model_name}:")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Precision | {prec:.4f} |")
    print(f"| Recall | {rec:.4f} |")
    print(f"| F1 Score | {f1:.4f} |")
    print(f"| PR-AUC | {pr_auc:.4f} |")
    print(f"| ROC-AUC | {roc_auc:.4f} |")
    print(f"| MCC | {mcc:.4f} |")
    
    return {
        "model": model_name,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "mcc": float(mcc)
    }

def evaluate_all_models(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluates all trained models, prints a sorted comparison table, and logs to MLflow.

    Args:
        trained_models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        DataFrame containing metrics for all models.
    """
    metrics_list = []
    for name, model in trained_models.items():
        y_pred, y_scores = get_predictions(model, X_test, name)
        m = compute_metrics(y_test, y_pred, y_scores, name)
        metrics_list.append(m)
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"evaluation_{name}", nested=True):
            for k, v in m.items():
                if k != "model":
                    mlflow.log_metric(k, v)
                    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.sort_values(by="pr_auc", ascending=False)
    print("\n--- Model Comparison Table ---")
    print(df_metrics.to_string(index=False))
    return df_metrics

def plot_pr_curves(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Plots precision-recall curves for all models on one figure.

    Args:
        trained_models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.
    """
    plt.figure(figsize=(10, 7))
    for name, model in trained_models.items():
        _, y_scores = get_predictions(model, X_test, name)
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    plt.title("Precision-Recall Curves - Baseline Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    os.makedirs("notebooks/figures", exist_ok=True)
    plt.savefig("notebooks/figures/pr_curves.png", dpi=300)
    plt.close()

def plot_confusion_matrices(trained_models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Plots confusion matrices for all 5 models in a 1x5 grid.

    Args:
        trained_models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred, _ = get_predictions(model, X_test, name)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    os.makedirs("notebooks/figures", exist_ok=True)
    plt.savefig("notebooks/figures/confusion_matrices.png", dpi=300)
    plt.close()
