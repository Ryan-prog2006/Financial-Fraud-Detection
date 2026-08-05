"""Evaluation functions and comparative reports for FinShield Deep Learning models."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef
)

def get_deep_predictions(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Runs model evaluation and returns true labels and probability scores.

    Args:
        model: PyTorch model.
        loader: Evaluation DataLoader.
        device: CPU or GPU device.

    Returns:
        A tuple of (y_true, y_scores).
    """
    model.eval()
    model.to(device)
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    logits_arr = np.array(all_logits)
    # Apply sigmoid activation to get probability scores
    y_scores = 1.0 / (1.0 + np.exp(-logits_arr))
    return np.array(all_targets), y_scores

def compute_deep_metrics(y_true: np.ndarray, y_scores: np.ndarray, model_name: str, threshold: float = 0.5) -> dict:
    """Computes standard evaluation metrics for deep learning models.

    Args:
        y_true: Test target labels.
        y_scores: Predicted probabilities.
        model_name: Name of the model.
        threshold: Confidence threshold for predictions.

    Returns:
        Dictionary of calculated metrics.
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    mcc = matthews_corrcoef(y_true, y_pred)
    
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

def compare_with_classical(deep_results: dict) -> None:
    """Prints a comparative side-by-side table of classical and deep learning models.

    Args:
        deep_results: Dictionary of metrics dictionaries for deep learning models.
    """
    print("\n=== Model Comparison: Classical vs Deep Learning ===")
    print(f"{'Model':<25} | {'PR-AUC':<8} | {'F1':<8}")
    print(f"{'-'*25}-|-{'-'*8}-|-{'-'*8}")
    print(f"{'Ensemble (Classical)':<25} | {0.8756:<8.4f} | {0.8705:<8.4f}")
    
    for name, metrics in deep_results.items():
        print(f"{name:<25} | {metrics['pr_auc']:<8.4f} | {metrics['f1']:<8.4f}")
        
    all_models = [("Ensemble (Classical)", 0.8756, 0.8705)]
    for name, metrics in deep_results.items():
        all_models.append((name, metrics["pr_auc"], metrics["f1"]))
        
    all_models.sort(key=lambda x: x[1], reverse=True)
    best_name, best_pr, _ = all_models[0]
    
    best_deep = max([metrics["pr_auc"] for metrics in deep_results.values()])
    diff = best_deep - 0.8756
    print(f"\nBest Overall Model: {best_name} with PR-AUC = {best_pr:.4f}")
    if diff > 0:
        print(f"Deep learning improved PR-AUC by +{diff:.4f} (+{diff*100/0.8756:.2f}%) over classical.")
    else:
        print(f"Classical ensemble remains ahead by {-diff:.4f} ({-diff*100/0.8756:.2f}%).")
