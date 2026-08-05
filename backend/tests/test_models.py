"""Unit tests for Phase 2 model training, evaluation, and predictions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.models.train import load_and_split, apply_smote, PROCESSED_PATH
from src.models.evaluate import get_predictions, compute_metrics

def test_load_and_split_stratified() -> None:
    """Test that train/test split retains identical class proportions within 0.05%."""
    X_train, X_test, y_train, y_test = load_and_split(PROCESSED_PATH)
    original_df = pd.read_csv(PROCESSED_PATH)
    
    original_fraud_rate = original_df["Class"].mean()
    test_fraud_rate = y_test.mean()
    
    # Assert within 0.05% (0.0005)
    assert abs(test_fraud_rate - original_fraud_rate) <= 0.0005

def test_smote_balances_classes() -> None:
    """Test that SMOTE accurately balances classes to a 50/50 ratio."""
    # Create an imbalanced dummy set with exactly 10 positive samples
    # to satisfy the default 5-neighbors requirement of SMOTE
    X_dummy = pd.DataFrame(np.random.normal(size=(100, 5)))
    y_dummy = pd.Series([0] * 90 + [1] * 10)
    
    X_res, y_res = apply_smote(X_dummy, y_dummy)
    counts = y_res.value_counts()
    
    assert counts[0] == counts[1]

def test_isolation_forest_prediction_binary() -> None:
    """Test that Isolation Forest output predictions are mapped to binary 0 and 1."""
    X_dummy = pd.DataFrame(np.random.normal(size=(50, 5)))
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_dummy)
    
    y_pred, y_scores = get_predictions(model, X_dummy, "isolation_forest")
    
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert len(y_scores) == len(X_dummy)

def test_metrics_keys_present() -> None:
    """Test that metric dictionary contains all the required keys."""
    y_test = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_scores = np.array([0.1, 0.9, 0.2, 0.3])
    
    metrics = compute_metrics(y_test, y_pred, y_scores, "test_model")
    expected_keys = {"model", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc"}
    
    assert expected_keys.issubset(metrics.keys())
