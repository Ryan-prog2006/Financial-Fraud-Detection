"""Evidently AI drift detection for FinShield production monitoring.

Detects data distribution shifts between the training reference data and
incoming production batches. Generates HTML reports and returns structured
drift summaries that feed the automated retraining pipeline.

Uses the Evidently legacy API (evidently>=0.7.x).
"""

from __future__ import annotations

import os
import random
from typing import Literal

import numpy as np
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFERENCE_DATA_PATH: str = "data/processed/creditcard_processed.csv"
DRIFT_REPORT_DIR: str = "data/drift_reports/"
PSI_THRESHOLD: float = 0.2
DRIFT_SHARE_THRESHOLD: float = 0.3

DriftType = Literal["none", "amount_drift", "night_drift", "feature_drift"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_data(path: str = REFERENCE_DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset and return the first 70% as reference data.

    This simulates the training-time distribution against which incoming
    production batches are compared.

    Args:
        path: Path to the processed creditcard CSV file.

    Returns:
        DataFrame containing the reference rows (70% of dataset).

    Raises:
        FileNotFoundError: If the processed data file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at '{path}'. "
            "Run the data pipeline (src/data/pipeline.py) first."
        )
    df = pd.read_csv(path)
    split_idx = int(len(df) * 0.70)
    reference_df = df.iloc[:split_idx].copy()
    print(
        f"Reference data loaded: {len(reference_df):,} rows "
        f"({len(reference_df) / len(df):.0%} of full dataset)"
    )
    return reference_df


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_new_batch(
    reference_df: pd.DataFrame,
    drift_type: DriftType = "none",
) -> pd.DataFrame:
    """Return the last 30% of the dataset as a simulated production batch.

    Optionally injects controlled drift to validate the detection pipeline.

    Drift injection modes:
        - ``"none"``: Returns data as-is — baseline comparison.
        - ``"amount_drift"``: Multiplies ``Amount_zscore`` by 1.5, simulating
          transactions becoming larger on average (e.g. holiday season).
        - ``"night_drift"``: Sets ``Is_night = 1`` for 40% of rows, simulating
          a shift toward nocturnal fraud activity.
        - ``"feature_drift"``: Adds Gaussian noise (σ=0.5) to ``V14`` and
          ``V10``, simulating a new fraud pattern in key discriminative features.

    Args:
        reference_df: The reference DataFrame (used to derive the full dataset
            path for loading the current batch slice).
        drift_type: One of ``"none"``, ``"amount_drift"``, ``"night_drift"``,
            ``"feature_drift"``.

    Returns:
        DataFrame of the simulated production batch with optional drift injected.
    """
    full_df = pd.read_csv(REFERENCE_DATA_PATH)
    split_idx = int(len(full_df) * 0.70)
    current_df = full_df.iloc[split_idx:].copy()

    if drift_type == "amount_drift":
        # Additive shift guarantees mean increases regardless of z-score sign.
        # +0.5 σ simulates transactions becoming larger on average (e.g. holiday season).
        current_df["Amount_zscore"] = current_df["Amount_zscore"] + 0.5

    elif drift_type == "night_drift":
        n_rows = len(current_df)
        night_mask = np.random.choice(
            [True, False], size=n_rows, p=[0.40, 0.60]
        )
        current_df.loc[night_mask, "Is_night"] = 1

    elif drift_type == "feature_drift":
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(loc=0.0, scale=0.5, size=len(current_df))
        current_df["V14"] = current_df["V14"] + noise
        current_df["V10"] = current_df["V10"] + noise

    elif drift_type != "none":
        raise ValueError(
            f"Unknown drift_type='{drift_type}'. "
            "Choose from: 'none', 'amount_drift', 'night_drift', 'feature_drift'."
        )

    print(
        f"Simulated batch: {len(current_df):,} rows, drift_type={drift_type}"
    )
    return current_df


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str = "drift_report",
) -> dict:
    """Run Evidently drift detection and save an HTML report.

    Builds a ``DataDriftPreset`` + ``DataQualityPreset`` Evidently report,
    extracts per-feature and dataset-level drift results, and serialises the
    report as an HTML file for browser review.

    Args:
        reference_df: Reference (training-time) DataFrame.
        current_df: Current (production) DataFrame.
        report_name: Filename stem for the saved HTML report (no extension).

    Returns:
        Dictionary with keys:
            - ``drift_share`` (float): Fraction of features that drifted.
            - ``drifted_features`` (list[str]): Names of drifted columns.
            - ``dataset_drift`` (bool): Overall dataset drift flag.
            - ``report_path`` (str): Absolute path to the saved HTML file.
    """
    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)

    # Drop target column so we compare features only
    ref = reference_df.drop(columns=["Class"], errors="ignore")
    cur = current_df.drop(columns=["Class"], errors="ignore")

    # Align columns (intersection) in case of any mismatch
    common_cols = list(ref.columns.intersection(cur.columns))
    ref = ref[common_cols]
    cur = cur[common_cols]

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)

    # Persist HTML report
    report_path = os.path.join(DRIFT_REPORT_DIR, f"{report_name}.html")
    report.save_html(report_path)

    # Parse structured results
    report_dict = report.as_dict()
    metrics = report_dict["metrics"]

    # DatasetDriftMetric — overall drift share
    dataset_metric = next(
        (m for m in metrics if m["metric"] == "DatasetDriftMetric"), None
    )
    drift_share: float = 0.0
    dataset_drift: bool = False
    if dataset_metric:
        result = dataset_metric["result"]
        drift_share = result.get("share_of_drifted_columns", 0.0)
        dataset_drift = result.get("dataset_drift", False)

    # DataDriftTable — per-feature drift flags
    drift_table = next(
        (m for m in metrics if m["metric"] == "DataDriftTable"), None
    )
    drifted_features: list[str] = []
    total_features: int = len(common_cols)
    if drift_table:
        drift_by_columns = drift_table["result"].get("drift_by_columns", {})
        drifted_features = [
            col
            for col, info in drift_by_columns.items()
            if info.get("drift_detected", False)
        ]

    print("=== Drift Detection Report ===")
    print(
        f"Features drifted: {len(drifted_features)}/{total_features} "
        f"({drift_share:.1%})"
    )
    print(f"Drifted features: {drifted_features}")
    print(f"Dataset drift detected: {'yes' if dataset_drift else 'no'}")
    print(f"Report saved to: {os.path.abspath(report_path)}")

    return {
        "drift_share": drift_share,
        "drifted_features": drifted_features,
        "dataset_drift": dataset_drift,
        "report_path": os.path.abspath(report_path),
    }


# ---------------------------------------------------------------------------
# Retraining decision
# ---------------------------------------------------------------------------

def should_retrain(drift_results: dict) -> bool:
    """Decide whether retraining should be triggered based on drift results.

    Retraining is recommended when the share of drifted features exceeds
    ``DRIFT_SHARE_THRESHOLD`` (default 30%).

    Args:
        drift_results: Dictionary returned by ``run_drift_detection()``.

    Returns:
        ``True`` if retraining is recommended, ``False`` otherwise.
    """
    drift_share: float = drift_results.get("drift_share", 0.0)
    retrain: bool = drift_share > DRIFT_SHARE_THRESHOLD
    print(
        f"Retraining needed: {retrain} "
        f"(drift share: {drift_share:.1%}, "
        f"threshold: {DRIFT_SHARE_THRESHOLD:.1%})"
    )
    return retrain
