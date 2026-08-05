"""Automated retraining pipeline for FinShield fraud detection models.

Triggered by the Prefect orchestration flow when drift is detected.
Retrains XGBoost on the latest processed data, logs the new run to MLflow,
and conditionally promotes to Staging/Production via the model registry.
"""

from __future__ import annotations

import datetime
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.mlops.registry import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    promote_to_staging,
    register_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFERENCE_DATA_PATH: str = "data/processed/creditcard_processed.csv"
RETRAINED_MODEL_PATH: str = "data/models/xgboost_retrained.pkl"

# Best params from Phase 2 Optuna tuning — used as hardcoded defaults
# so retraining doesn't require another full Optuna search.
BEST_XGB_PARAMS: dict = {
    "n_estimators": 429,
    "max_depth": 5,
    "learning_rate": 0.08328453326222358,
    "subsample": 0.7638667196069421,
    "colsample_bytree": 0.9066256620603783,
    "scale_pos_weight": 577,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}

# Phase 2 production baseline — compared against new model
BASELINE_PR_AUC: float = 0.8725


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def retrain_model(
    data_path: str = REFERENCE_DATA_PATH,
    trigger_reason: str = "scheduled",
) -> dict:
    """Retrain XGBoost on the latest processed data and log to MLflow.

    Pipeline:
        1. Load processed data (stratified 50% sample for speed on CPU).
        2. Stratified train/test split (80/20, random_state=42).
        3. Apply SMOTE to the training set.
        4. Train XGBoost with Phase-2-tuned hyperparameters.
        5. Evaluate on the held-out test set (PR-AUC, F1).
        6. Save model pickle to ``RETRAINED_MODEL_PATH``.
        7. Log run to MLflow with drift/trigger metadata tags.

    Args:
        data_path: Path to the processed CSV file.
        trigger_reason: Human-readable trigger label (e.g. ``"drift_detected"``,
            ``"scheduled"``, ``"manual"``).

    Returns:
        Metrics dictionary with keys: ``pr_auc``, ``f1``, ``precision``,
        ``recall``, ``improvement``.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at '{data_path}'.")

    today = datetime.date.today().isoformat()

    # ------------------------------------------------------------------
    # 1. Load + stratified subsample (50%) for fast CPU retraining
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)
    df_sample, _ = train_test_split(
        df, test_size=0.50, stratify=df["Class"], random_state=42
    )
    print(
        f"Retraining on {len(df_sample):,} rows "
        f"(50% stratified sample of {len(df):,})"
    )

    X = df_sample.drop(columns=["Class"])
    y = df_sample["Class"]

    # ------------------------------------------------------------------
    # 2. Stratified train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # ------------------------------------------------------------------
    # 3. SMOTE
    # ------------------------------------------------------------------
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Train XGBoost with Phase-2 best params
    # ------------------------------------------------------------------
    model = XGBClassifier(**BEST_XGB_PARAMS, verbosity=0)
    model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=False)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_test, y_scores))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    precision = float(
        np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)
    )
    recall = float(
        np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
    )
    improvement = pr_auc - BASELINE_PR_AUC

    metrics = {
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "improvement": improvement,
    }

    # ------------------------------------------------------------------
    # 6. Save model pickle
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(RETRAINED_MODEL_PATH), exist_ok=True)
    with open(RETRAINED_MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)

    # ------------------------------------------------------------------
    # 7. Log to MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"retrain_{today}"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.set_tags(
            {
                "trigger": trigger_reason,
                "retrain_date": today,
                "auto_retrained": "true",
                "sample_fraction": "0.50",
            }
        )
        mlflow.sklearn.log_model(model, artifact_path="model")

    sign = "+" if improvement >= 0 else ""
    print("=== Retraining Complete ===")
    print(f"Trigger:                   {trigger_reason}")
    print(f"New model PR-AUC:          {pr_auc:.4f}")
    print(f"Previous production PR-AUC:{BASELINE_PR_AUC:.4f}")
    print(f"Improvement:               {sign}{improvement:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------

def compare_and_promote(
    new_metrics: dict,
    production_threshold: float = 0.85,
) -> bool:
    """Decide whether to promote the retrained model to Production.

    Compares the new model's PR-AUC against *production_threshold*.
    If the threshold is met, the model is registered and promoted to Staging.
    Full Production promotion is intentionally left to a human reviewer in
    a real deployment; in this demo pipeline it is automated.

    Args:
        new_metrics: Metrics dict returned by ``retrain_model()``.
        production_threshold: Minimum PR-AUC for promotion (default 0.85).

    Returns:
        ``True`` if the model was promoted, ``False`` otherwise.
    """
    new_pr_auc = new_metrics.get("pr_auc", 0.0)
    promoted = new_pr_auc >= production_threshold

    print("\n=== Promotion Decision ===")
    print(f"New model PR-AUC:     {new_pr_auc:.4f}")
    print(f"Production threshold: {production_threshold:.4f}")

    if promoted:
        print(
            f"✅  New model meets threshold — registering and promoting to Staging."
        )
        try:
            run_id = register_model(
                model_path=RETRAINED_MODEL_PATH,
                model_name="xgboost_retrained",
                metrics=new_metrics,
                run_name="retrained_promoted",
            )
            promote_to_staging(run_id)
        except Exception as exc:
            print(f"⚠️  Registry promotion failed (non-fatal): {exc}")
    else:
        delta = production_threshold - new_pr_auc
        print(
            f"❌  New model does not meet threshold "
            f"(gap: {delta:.4f}) — keeping current production model."
        )

    return promoted
