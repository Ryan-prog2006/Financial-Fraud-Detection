"""Production performance monitoring for FinShield fraud detection.

Simulates 7 days of production batches, computes per-batch metrics,
plots a multi-panel monitoring dashboard, and generates a summary
recommendation for the operations team.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH: str = "data/processed/creditcard_processed.csv"
FIGURES_DIR: str = "notebooks/figures/"

# Alert thresholds
PR_AUC_THRESHOLD: float = 0.80
F1_THRESHOLD: float = 0.75


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_production_batches(n_batches: int = 7) -> list[dict]:
    """Simulate *n_batches* days of production monitoring.

    Each day draws 5,000 transactions with replacement, adds slight noise to
    simulate real distribution shift, runs model predictions, and applies a
    random degradation multiplier to mimic gradual model decay.

    Args:
        n_batches: Number of daily batches to simulate (default 7).

    Returns:
        List of metric dictionaries, one per batch, with keys:
        ``day``, ``pr_auc``, ``precision``, ``recall``, ``f1``.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Load production model (XGBoost tuned from Phase 2)
    try:
        import pickle
        with open("data/models/xgboost_tuned.pkl", "rb") as fh:
            model = pickle.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Production model not found at 'data/models/xgboost_tuned.pkl'. "
            "Ensure Phase 2 pipeline has been executed."
        )

    rng = np.random.default_rng(seed=0)
    results: list[dict] = []

    for day in range(1, n_batches + 1):
        # Sample 5,000 rows with replacement
        idx = rng.choice(len(X), size=5_000, replace=True)
        X_batch = X.iloc[idx].copy()
        y_batch = y.iloc[idx].copy()

        # Add slight Gaussian noise to simulate distribution drift
        noise_cols = [c for c in X_batch.columns if c.startswith("V")]
        noise = rng.normal(
            loc=0.0, scale=0.02 * day, size=(len(X_batch), len(noise_cols))
        )
        X_batch[noise_cols] += noise

        # Predict
        y_scores = model.predict_proba(X_batch)[:, 1]
        y_pred = (y_scores >= 0.5).astype(int)

        # Compute metrics
        pr_auc = float(average_precision_score(y_batch, y_scores))
        prec = float(precision_score(y_batch, y_pred, zero_division=0))
        rec = float(recall_score(y_batch, y_pred, zero_division=0))
        f1 = float(f1_score(y_batch, y_pred, zero_division=0))

        # Simulate slight degradation (±2% random walk around true value)
        degradation = 0.97 + rng.random() * 0.04
        pr_auc = float(np.clip(pr_auc * degradation, 0.0, 1.0))

        results.append(
            {
                "day": day,
                "pr_auc": round(pr_auc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_performance_over_time(batch_results: list[dict]) -> None:
    """Plot a 2×2 monitoring dashboard of model performance over time.

    Each panel shows one metric (PR-AUC, Precision, Recall, F1) plotted as a
    line with markers. A red dashed horizontal line marks the alert threshold
    for PR-AUC and F1. Any metric that drops below its threshold triggers an
    alert printed to stdout.

    Saves to ``notebooks/figures/monitoring_dashboard.png``.

    Args:
        batch_results: List of batch metric dicts from
            ``simulate_production_batches()``.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    days = [r["day"] for r in batch_results]
    metrics = {
        "PR-AUC": [r["pr_auc"] for r in batch_results],
        "Precision": [r["precision"] for r in batch_results],
        "Recall": [r["recall"] for r in batch_results],
        "F1 Score": [r["f1"] for r in batch_results],
    }
    thresholds = {
        "PR-AUC": PR_AUC_THRESHOLD,
        "F1 Score": F1_THRESHOLD,
    }
    colors = {
        "PR-AUC": "#4C9BE8",
        "Precision": "#27AE60",
        "Recall": "#E67E22",
        "F1 Score": "#9B59B6",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "FinShield Model Performance Monitoring — 7 Day Simulation",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    axes_flat = axes.flatten()

    for ax, (metric_name, values) in zip(axes_flat, metrics.items()):
        ax.plot(
            days,
            values,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=colors[metric_name],
            label=metric_name,
        )
        ax.fill_between(days, values, alpha=0.12, color=colors[metric_name])

        # Alert threshold line
        if metric_name in thresholds:
            threshold = thresholds[metric_name]
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"Threshold ({threshold})",
            )
            # Check for breaches
            for d, v in zip(days, values):
                if v < threshold:
                    print(
                        f"⚠️  ALERT: {metric_name} dropped below threshold "
                        f"on day {d} ({v:.4f} < {threshold})"
                    )

        ax.set_title(metric_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel(metric_name)
        ax.set_xticks(days)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "monitoring_dashboard.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Monitoring dashboard saved to: {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_monitoring_summary(batch_results: list[dict]) -> dict:
    """Generate a structured summary of the 7-day monitoring window.

    Recommendation logic:
        - avg_pr_auc > 0.85  → ``"stable"``
        - 0.80 < avg_pr_auc ≤ 0.85 → ``"monitor_closely"``
        - avg_pr_auc ≤ 0.80  → ``"retrain_now"``

    Args:
        batch_results: List of batch metric dicts from
            ``simulate_production_batches()``.

    Returns:
        Dictionary with keys: ``avg_pr_auc``, ``min_pr_auc``, ``avg_f1``,
        ``min_f1``, ``alerts_triggered``, ``recommendation``.
    """
    pr_aucs = [r["pr_auc"] for r in batch_results]
    f1s = [r["f1"] for r in batch_results]

    avg_pr_auc = float(np.mean(pr_aucs))
    min_pr_auc = float(np.min(pr_aucs))
    avg_f1 = float(np.mean(f1s))
    min_f1 = float(np.min(f1s))

    alerts: list[str] = []
    for r in batch_results:
        if r["pr_auc"] < PR_AUC_THRESHOLD:
            alerts.append(
                f"Day {r['day']}: PR-AUC={r['pr_auc']:.4f} below {PR_AUC_THRESHOLD}"
            )
        if r["f1"] < F1_THRESHOLD:
            alerts.append(
                f"Day {r['day']}: F1={r['f1']:.4f} below {F1_THRESHOLD}"
            )

    if avg_pr_auc > 0.85:
        recommendation = "stable"
    elif avg_pr_auc > 0.80:
        recommendation = "monitor_closely"
    else:
        recommendation = "retrain_now"

    return {
        "avg_pr_auc": round(avg_pr_auc, 4),
        "min_pr_auc": round(min_pr_auc, 4),
        "avg_f1": round(avg_f1, 4),
        "min_f1": round(min_f1, 4),
        "alerts_triggered": alerts,
        "recommendation": recommendation,
    }
