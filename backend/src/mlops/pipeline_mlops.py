"""Master orchestrator for FinShield Phase 5 MLOps pipeline.

Runs the full MLOps loop using Prefect 3.x:
  1. Register the Phase 2 XGBoost tuned model in MLflow Model Registry
  2. Promote through Staging → Production
  3. List all registered model versions
  4. Run drift detection (no-drift baseline)
  5. Run drift detection (amount_drift — triggers retraining)
  6. Retrain model with trigger_reason="drift_detected"
  7. Compare and conditionally promote new model
  8. Simulate 7 days of production monitoring
  9. Plot monitoring dashboard
  10. Print Phase 5 summary

Usage:
    python src/mlops/pipeline_mlops.py
"""

from __future__ import annotations

from prefect import flow, task

from src.mlops.registry import (
    REGISTERED_MODEL_NAME,
    list_model_versions,
    promote_to_production,
    promote_to_staging,
    register_model,
)
from src.mlops.drift_detector import (
    load_reference_data,
    run_drift_detection,
    should_retrain,
    simulate_new_batch,
)
from src.mlops.retrain import compare_and_promote, retrain_model
from src.mlops.monitor import (
    generate_monitoring_summary,
    plot_performance_over_time,
    simulate_production_batches,
)

# ---------------------------------------------------------------------------
# Phase 2 baseline metrics (XGBoost tuned)
# ---------------------------------------------------------------------------
XGBOOST_TUNED_METRICS: dict = {
    "pr_auc": 0.8725,
    "f1": 0.8300,
    "precision": 0.8642,
    "recall": 0.7959,
    "roc_auc": 0.9741,
    "mcc": 0.8291,
}


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------

@task(name="register-baseline-model", retries=2, retry_delay_seconds=5)
def task_register_model() -> str:
    """Register the Phase 2 XGBoost tuned model in MLflow Model Registry."""
    print("\n" + "=" * 60)
    print("STEP 1: Registering Phase 2 XGBoost tuned model")
    print("=" * 60)
    run_id = register_model(
        model_path="data/models/xgboost_tuned.pkl",
        model_name="xgboost_tuned",
        metrics=XGBOOST_TUNED_METRICS,
        run_name="xgboost_tuned_phase2_registration",
    )
    return run_id


@task(name="promote-to-staging", retries=2, retry_delay_seconds=5)
def task_promote_staging(run_id: str) -> str:
    """Promote the registered model version to Staging."""
    print("\n" + "=" * 60)
    print("STEP 2a: Promoting to Staging")
    print("=" * 60)
    promote_to_staging(run_id)
    return run_id


@task(name="promote-to-production", retries=2, retry_delay_seconds=5)
def task_promote_production() -> None:
    """Promote version 1 to Production (first registered version)."""
    print("\n" + "=" * 60)
    print("STEP 2b: Promoting version 1 to Production")
    print("=" * 60)
    promote_to_production(version=1)


@task(name="list-model-versions", retries=1, retry_delay_seconds=5)
def task_list_versions() -> None:
    """Print a table of all registered model versions."""
    print("\n" + "=" * 60)
    print("STEP 3: Model Registry Version Table")
    print("=" * 60)
    list_model_versions()


@task(name="drift-check-no-drift", retries=1, retry_delay_seconds=5)
def task_drift_no_drift() -> dict:
    """Run drift detection on baseline data — expect no retraining needed."""
    print("\n" + "=" * 60)
    print("STEP 4: Drift Check — no drift injected (baseline)")
    print("=" * 60)
    reference_df = load_reference_data()
    current_df = simulate_new_batch(reference_df, drift_type="none")
    drift_results = run_drift_detection(
        reference_df, current_df, report_name="drift_report_no_drift"
    )
    retrain_needed = should_retrain(drift_results)
    return drift_results


@task(name="drift-check-amount-drift", retries=1, retry_delay_seconds=5)
def task_drift_amount() -> dict:
    """Run drift detection with amount_drift — expect retraining triggered."""
    print("\n" + "=" * 60)
    print("STEP 5: Drift Check — amount_drift injected")
    print("=" * 60)
    reference_df = load_reference_data()
    current_df = simulate_new_batch(reference_df, drift_type="amount_drift")
    drift_results = run_drift_detection(
        reference_df, current_df, report_name="drift_report_amount_drift"
    )
    retrain_needed = should_retrain(drift_results)
    return drift_results


@task(name="retrain-model", retries=1, retry_delay_seconds=10)
def task_retrain(drift_results: dict) -> dict:
    """Retrain XGBoost with trigger_reason='drift_detected'."""
    print("\n" + "=" * 60)
    print("STEP 6: Automated Retraining")
    print("=" * 60)
    new_metrics = retrain_model(trigger_reason="drift_detected")
    return new_metrics


@task(name="compare-and-promote", retries=1, retry_delay_seconds=5)
def task_compare_promote(new_metrics: dict) -> bool:
    """Compare new model metrics and conditionally promote."""
    print("\n" + "=" * 60)
    print("STEP 7: Compare and Promote")
    print("=" * 60)
    return compare_and_promote(new_metrics)


@task(name="simulate-monitoring", retries=1, retry_delay_seconds=5)
def task_simulate_monitoring() -> list[dict]:
    """Simulate 7 days of production batch monitoring."""
    print("\n" + "=" * 60)
    print("STEP 8: 7-Day Production Monitoring Simulation")
    print("=" * 60)
    return simulate_production_batches(n_batches=7)


@task(name="plot-monitoring-dashboard", retries=1, retry_delay_seconds=5)
def task_plot_dashboard(batch_results: list[dict]) -> dict:
    """Plot monitoring dashboard and generate summary."""
    print("\n" + "=" * 60)
    print("STEP 9: Plotting Monitoring Dashboard")
    print("=" * 60)
    plot_performance_over_time(batch_results)
    summary = generate_monitoring_summary(batch_results)
    return summary


# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------

@flow(name="finshield-mlops-pipeline", log_prints=True)
def mlops_pipeline() -> None:
    """End-to-end FinShield Phase 5 MLOps orchestration flow."""

    # 1. Register baseline model
    run_id = task_register_model()

    # 2. Promote Staging → Production
    task_promote_staging(run_id)
    task_promote_production()

    # 3. Version table
    task_list_versions()

    # 4. Drift check — no drift
    drift_no_drift = task_drift_no_drift()

    # 5. Drift check — amount drift
    drift_amount = task_drift_amount()

    # 6. Retrain (triggered by amount drift)
    new_metrics = task_retrain(drift_amount)

    # 7. Compare and conditionally promote
    promoted = task_compare_promote(new_metrics)

    # 8. 7-day monitoring simulation
    batch_results = task_simulate_monitoring()

    # 9. Plot dashboard + get summary
    summary = task_plot_dashboard(batch_results)

    # 10. Final summary
    _print_phase5_summary(promoted, summary)


def _print_phase5_summary(promoted: bool, summary: dict) -> None:
    """Print the Phase 5 completion summary block."""
    print("\n" + "=" * 60)
    print("=== Phase 5 MLOps Summary ===")
    print("=" * 60)
    print(f"Model in production:       {REGISTERED_MODEL_NAME} version 1")
    print(f"Drift detection:           operational (HTML reports saved to data/drift_reports/)")
    print(f"Auto-retraining:           {'triggered and completed' if promoted else 'triggered — model did not meet promotion threshold'}")
    print(f"7-day monitoring:          {summary['recommendation']}")
    print(f"  avg PR-AUC:              {summary['avg_pr_auc']:.4f}")
    print(f"  min PR-AUC:              {summary['min_pr_auc']:.4f}")
    print(f"  avg F1:                  {summary['avg_f1']:.4f}")
    if summary["alerts_triggered"]:
        print(f"  alerts:                  {len(summary['alerts_triggered'])}")
        for alert in summary["alerts_triggered"]:
            print(f"    ⚠️  {alert}")
    else:
        print(f"  alerts:                  none")
    print(f"Monitoring chart:          notebooks/figures/monitoring_dashboard.png")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mlops_pipeline()
