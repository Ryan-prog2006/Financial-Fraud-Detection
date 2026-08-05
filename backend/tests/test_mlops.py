"""Pytest unit tests for FinShield Phase 5 MLOps components.

Tests cover drift simulation, retraining decision logic, monitoring
summary generation, and the existence of required deployment artefacts.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_df() -> pd.DataFrame:
    """Load the reference data slice once for all tests in this module."""
    from src.mlops.drift_detector import load_reference_data
    return load_reference_data()


# ---------------------------------------------------------------------------
# Drift simulation tests
# ---------------------------------------------------------------------------

def test_simulate_new_batch_no_drift(reference_df: pd.DataFrame) -> None:
    """Batch with drift_type='none' must have the same columns as reference."""
    from src.mlops.drift_detector import simulate_new_batch

    batch = simulate_new_batch(reference_df, drift_type="none")

    assert isinstance(batch, pd.DataFrame), "Expected a DataFrame"
    # Every feature column in reference should exist in the batch
    ref_cols = set(reference_df.columns)
    batch_cols = set(batch.columns)
    assert ref_cols == batch_cols, (
        f"Column mismatch.\nOnly in reference: {ref_cols - batch_cols}\n"
        f"Only in batch: {batch_cols - ref_cols}"
    )


def test_simulate_new_batch_amount_drift(reference_df: pd.DataFrame) -> None:
    """Amount drift must produce a higher mean Amount_zscore than the undrifted batch."""
    from src.mlops.drift_detector import simulate_new_batch

    no_drift_batch = simulate_new_batch(reference_df, drift_type="none")
    drifted_batch = simulate_new_batch(reference_df, drift_type="amount_drift")

    no_drift_mean = no_drift_batch["Amount_zscore"].mean()
    drifted_mean = drifted_batch["Amount_zscore"].mean()

    assert drifted_mean > no_drift_mean, (
        f"Expected drifted mean ({drifted_mean:.4f}) > no-drift mean ({no_drift_mean:.4f})"
    )


# ---------------------------------------------------------------------------
# Retraining decision tests
# ---------------------------------------------------------------------------

def test_should_retrain_true() -> None:
    """should_retrain returns True when drift_share exceeds the threshold."""
    from src.mlops.drift_detector import should_retrain

    high_drift = {"drift_share": 0.4, "drifted_features": [], "dataset_drift": True}
    result = should_retrain(high_drift)
    assert result is True, "Expected should_retrain to return True for drift_share=0.4"


def test_should_retrain_false() -> None:
    """should_retrain returns False when drift_share is below the threshold."""
    from src.mlops.drift_detector import should_retrain

    low_drift = {"drift_share": 0.1, "drifted_features": [], "dataset_drift": False}
    result = should_retrain(low_drift)
    assert result is False, "Expected should_retrain to return False for drift_share=0.1"


# ---------------------------------------------------------------------------
# Monitoring summary tests
# ---------------------------------------------------------------------------

def _make_batch_results(pr_auc_val: float) -> list[dict]:
    """Helper: build 7 identical batch result dicts with the given pr_auc."""
    return [
        {
            "day": d,
            "pr_auc": pr_auc_val,
            "precision": 0.90,
            "recall": 0.85,
            "f1": 0.87,
        }
        for d in range(1, 8)
    ]


def test_monitoring_summary_stable() -> None:
    """avg PR-AUC > 0.85 → recommendation must be 'stable'."""
    from src.mlops.monitor import generate_monitoring_summary

    results = _make_batch_results(pr_auc_val=0.90)
    summary = generate_monitoring_summary(results)

    assert summary["recommendation"] == "stable", (
        f"Expected 'stable', got '{summary['recommendation']}'"
    )
    assert abs(summary["avg_pr_auc"] - 0.90) < 1e-6


def test_monitoring_summary_retrain() -> None:
    """avg PR-AUC <= 0.80 → recommendation must be 'retrain_now'."""
    from src.mlops.monitor import generate_monitoring_summary

    results = _make_batch_results(pr_auc_val=0.75)
    summary = generate_monitoring_summary(results)

    assert summary["recommendation"] == "retrain_now", (
        f"Expected 'retrain_now', got '{summary['recommendation']}'"
    )
    assert abs(summary["avg_pr_auc"] - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# Deployment artefact existence tests
# ---------------------------------------------------------------------------

def test_docker_compose_exists() -> None:
    """docker-compose.yml must exist in the project root."""
    path = "docker-compose.yml"
    if not os.path.exists(path):
        path = "../docker-compose.yml"
    assert os.path.exists(path), (
        "docker-compose.yml not found in project root."
    )


def test_model_card_exists() -> None:
    """model_card.md must exist and be substantially populated (>500 chars)."""
    path = "model_card.md"
    if not os.path.exists(path):
        path = "../model_card.md"
    assert os.path.exists(path), "model_card.md not found in project root."

    size = os.path.getsize(path)
    assert size > 500, (
        f"model_card.md is too short ({size} bytes). Expected > 500 characters."
    )
