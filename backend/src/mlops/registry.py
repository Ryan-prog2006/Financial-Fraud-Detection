"""MLflow Model Registry utilities for FinShield production governance.

Handles model registration, stage promotion, production loading,
and version listing via the MLflow Model Registry API.
"""

from __future__ import annotations

import os
import pickle
import datetime
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = "mlruns"
EXPERIMENT_NAME: str = "finshield_classical_ml"
REGISTERED_MODEL_NAME: str = "finshield_fraud_detector"


class ModelNotFoundError(Exception):
    """Raised when no Production model exists in the registry."""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_model(
    model_path: str,
    model_name: str,
    metrics: dict,
    run_name: str,
) -> str:
    """Register a serialised sklearn model into the MLflow Model Registry.

    Loads the pickle from *model_path*, starts a new MLflow run, logs all
    provided metrics, logs the model artefact, and registers it under
    ``REGISTERED_MODEL_NAME``.

    Args:
        model_path: Absolute or relative path to the ``.pkl`` model file.
        model_name: Human-readable label used as the MLflow ``run_name``.
        metrics: Dictionary of metric names → float values to log.
        run_name: Display name for the MLflow run.

    Returns:
        The MLflow run ID string for the newly created run.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        run_id: str = run.info.run_id

        # Log metrics
        for metric_key, metric_val in metrics.items():
            mlflow.log_metric(metric_key, float(metric_val))

        # Tag with extra context
        mlflow.set_tags(
            {
                "model_name": model_name,
                "registered_by": "finshield_registry",
                "registration_date": datetime.date.today().isoformat(),
            }
        )

        # Log model artefact and register it
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    print(f"Model registered: {model_name} | Run ID: {run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Stage promotion
# ---------------------------------------------------------------------------

def promote_to_staging(run_id: str) -> None:
    """Transition the model version linked to *run_id* to the Staging stage.

    Args:
        run_id: MLflow run ID whose registered model version should be promoted.

    Raises:
        ModelNotFoundError: If no version is found for the given run ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    target = next((v for v in versions if v.run_id == run_id), None)

    if target is None:
        raise ModelNotFoundError(
            f"No registered version found for run_id='{run_id}'. "
            "Ensure register_model() was called first."
        )

    version_number = target.version
    pr_auc = _get_run_metric(run_id, "pr_auc")
    today = datetime.date.today().isoformat()

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version_number,
        stage="Staging",
        archive_existing_versions=False,
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version_number,
        description=(
            f"Promoted to Staging after evaluation. "
            f"PR-AUC: {pr_auc:.4f}. Date: {today}"
        ),
    )
    print(f"Model version {version_number} promoted to Staging")


def promote_to_production(version: int) -> None:
    """Promote *version* to Production, archiving the current Production model.

    Args:
        version: Integer version number to promote.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=str(version),
        description="Promoted to Production. Replacing previous version.",
    )
    print(
        f"Version {version} is now PRODUCTION. Previous version archived."
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def get_production_model():
    """Load and return the current Production model from the MLflow registry.

    Returns:
        The deserialised scikit-learn model object.

    Raises:
        ModelNotFoundError: If no version with stage ``Production`` exists.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        raise ModelNotFoundError(
            f"No Production model found for '{REGISTERED_MODEL_NAME}'. "
            "Run promote_to_production() first."
        )

    latest_prod = max(prod_versions, key=lambda v: int(v.version))
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)

    print(
        f"Loaded production model: {REGISTERED_MODEL_NAME} "
        f"version {latest_prod.version}"
    )
    return model


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_model_versions() -> pd.DataFrame:
    """Return a DataFrame of all registered model versions.

    Columns: version, stage, run_id, creation_time.

    Returns:
        Pandas DataFrame sorted by version descending.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    rows = []
    for v in versions:
        rows.append(
            {
                "version": int(v.version),
                "stage": v.current_stage,
                "run_id": v.run_id[:12] + "…",
                "creation_time": datetime.datetime.fromtimestamp(
                    v.creation_timestamp / 1000
                ).strftime("%Y-%m-%d %H:%M"),
            }
        )

    df = pd.DataFrame(rows).sort_values("version", ascending=False).reset_index(
        drop=True
    )
    print("\n=== Registered Model Versions ===")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_run_metric(run_id: str, metric_name: str, default: float = 0.0) -> float:
    """Retrieve a single metric value from an MLflow run.

    Args:
        run_id: MLflow run ID.
        metric_name: Name of the metric to retrieve.
        default: Value returned if the metric is not found.

    Returns:
        Float metric value.
    """
    try:
        client = MlflowClient()
        metric_history = client.get_metric_history(run_id, metric_name)
        if metric_history:
            return metric_history[-1].value
    except MlflowException:
        pass
    return default
