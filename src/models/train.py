"""Model training and resampling functions for FinShield classical ML models."""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Constants
PROCESSED_PATH: str = "data/processed/creditcard_processed.csv"
MODELS_DIR: str = "data/models/"
MLFLOW_EXPERIMENT: str = "finshield_classical_ml"

def load_and_split(processed_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads the processed data and performs a stratified train/test split.

    Args:
        processed_path: Path to the processed transaction CSV file.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found at: {processed_path}")
    df = pd.read_csv(processed_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Class distribution - Train:")
    print(y_train.value_counts(normalize=True))
    print("Class distribution - Test:")
    print(y_test.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Applies SMOTE to balance the training set classes.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target labels Series.

    Returns:
        A tuple of (X_resampled, y_resampled).
    """
    print("Class counts before SMOTE:")
    print(y_train.value_counts())
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print("Class counts after SMOTE:")
    print(y_res.value_counts())
    return X_res, y_res

def get_models() -> dict:
    """Returns a dictionary of 5 untrained model objects.

    Returns:
        A dictionary mapping model names to their estimators.
    """
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(scale_pos_weight=577, eval_metric="aucpr", random_state=42),
        "lightgbm": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
        "isolation_forest": IsolationForest(contamination=0.00173, random_state=42)
    }

def train_all_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_resampled: pd.DataFrame,
    y_resampled: pd.Series
) -> dict:
    """Trains all 5 baseline models and logs them to MLflow.

    Args:
        models: Dictionary of model estimators.
        X_train: Original training features.
        y_train: Original training labels.
        X_resampled: SMOTE-resampled training features.
        y_resampled: SMOTE-resampled training labels.

    Returns:
        A dictionary containing the trained model objects.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=f"baseline_{name}", nested=True):
            if name == "isolation_forest":
                model.fit(X_train)
            else:
                model.fit(X_resampled, y_resampled)
            
            # Save model to disk
            model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
                
            # Log model to MLflow
            mlflow.sklearn.log_model(model, artifact_path=name)
            mlflow.log_param("model_name", name)
            trained_models[name] = model
            
    return trained_models
