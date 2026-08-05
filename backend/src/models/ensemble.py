"""Ensembling functions for combining XGBoost and LightGBM model predictions."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

def build_ensemble(xgb_model, lgbm_model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Builds and evaluates 3 ensembling strategies and returns the best one.

    Args:
        xgb_model: Trained XGBoost model.
        lgbm_model: Trained LightGBM model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        A dictionary describing the best ensembling strategy.
    """
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    
    strategies = {
        "simple_average": (xgb_proba + lgbm_proba) / 2,
        "weighted_average": 0.6 * xgb_proba + 0.4 * lgbm_proba,
        "max_probability": np.maximum(xgb_proba, lgbm_proba)
    }
    
    results = []
    for name, scores in strategies.items():
        preds = (scores >= 0.5).astype(int)
        pr_auc = average_precision_score(y_test, scores)
        f1 = f1_score(y_test, preds, zero_division=0)
        results.append({"strategy_name": name, "pr_auc": pr_auc, "f1": f1, "y_scores": scores})
        
    df_res = pd.DataFrame(results)
    print("\n--- Ensemble Strategies Comparison ---")
    print(df_res[["strategy_name", "pr_auc", "f1"]].to_string(index=False))
    
    best_row = df_res.sort_values(by="pr_auc", ascending=False).iloc[0]
    best_strategy = dict(best_row)
    
    # Save the best predictions
    os.makedirs("data/models", exist_ok=True)
    np.save("data/models/ensemble_predictions.npy", best_strategy["y_scores"])
    print(f"Saved best strategy predictions ({best_strategy['strategy_name']}) to data/models/ensemble_predictions.npy")
    
    return best_strategy
