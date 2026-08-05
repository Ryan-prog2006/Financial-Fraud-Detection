"""Hyperparameter tuning using Optuna for XGBoost and LightGBM."""

import os
import pickle
import optuna
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow
import mlflow.sklearn

MODELS_DIR = "data/models/"

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Tunes XGBoost hyperparameters using Optuna.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dictionary of best parameters.
    """
    print("Tuning XGBoost...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": 577,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
        return scores.mean()
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print(f"XGBoost Best PR-AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return study.best_params

def tune_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Tunes LightGBM hyperparameters using Optuna.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dictionary of best parameters.
    """
    print("Tuning LightGBM...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        model = LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
        return scores.mean()
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print(f"LightGBM Best PR-AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return study.best_params

def retrain_with_best_params(
    X_resampled: pd.DataFrame,
    y_resampled: pd.Series,
    xgb_params: dict,
    lgbm_params: dict
) -> tuple[XGBClassifier, LGBMClassifier]:
    """Retrains models with tuned parameters on SMOTE resampled data.

    Args:
        X_resampled: Balanced training features.
        y_resampled: Balanced training labels.
        xgb_params: Best hyperparameters for XGBoost.
        lgbm_params: Best hyperparameters for LightGBM.

    Returns:
        A tuple of (xgb_tuned, lgbm_tuned) trained models.
    """
    print("Retraining XGBoost with best parameters...")
    xgb_model = XGBClassifier(**xgb_params, random_state=42, n_jobs=-1)
    xgb_model.fit(X_resampled, y_resampled)
    
    print("Retraining LightGBM with best parameters...")
    lgbm_model = LGBMClassifier(**lgbm_params, random_state=42, verbose=-1, n_jobs=-1)
    lgbm_model.fit(X_resampled, y_resampled)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in [("xgboost_tuned", xgb_model), ("lightgbm_tuned", lgbm_model)]:
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
        with mlflow.start_run(run_name=f"tuned_{name}", nested=True):
            mlflow.sklearn.log_model(model, artifact_path=name)
            mlflow.set_tag("tuned", "True")
            
    return xgb_model, lgbm_model
