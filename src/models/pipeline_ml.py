"""Master pipeline orchestrator for FinShield Phase 2 Classical ML models."""

import os
import mlflow
from src.models.train import (
    load_and_split,
    apply_smote,
    get_models,
    train_all_models,
    PROCESSED_PATH,
    MLFLOW_EXPERIMENT
)
from src.models.evaluate import (
    evaluate_all_models,
    plot_pr_curves,
    plot_confusion_matrices
)
from src.models.tune import (
    tune_xgboost,
    tune_lightgbm,
    retrain_with_best_params
)
from src.models.explain import (
    generate_shap_summary,
    explain_single_prediction
)
from src.models.ensemble import build_ensemble

def main() -> None:
    """Runs the end-to-end Classical ML pipeline."""
    print("Initializing Phase 2 ML Pipeline...")
    
    # Set the mlflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name="finshield_classical_ml_pipeline"):
        # 1. Load and split data
        X_train, X_test, y_train, y_test = load_and_split(PROCESSED_PATH)
        
        # 2. Apply SMOTE
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        
        # 3. Train all 5 baseline models
        models = get_models()
        trained_models = train_all_models(models, X_train, y_train, X_resampled, y_resampled)
        
        # 4. Evaluate all models
        print("Evaluating baseline models...")
        df_metrics = evaluate_all_models(trained_models, X_test, y_test)
        
        # Generate diagnostic plots for baseline models
        plot_pr_curves(trained_models, X_test, y_test)
        plot_confusion_matrices(trained_models, X_test, y_test)
        
        # 5. Tune XGBoost and LightGBM with Optuna
        xgb_best_params = tune_xgboost(X_train, y_train)
        lgbm_best_params = tune_lightgbm(X_train, y_train)
        
        # 6. Retrain with best params
        xgb_tuned, lgbm_tuned = retrain_with_best_params(
            X_resampled, y_resampled, xgb_best_params, lgbm_best_params
        )
        
        # 7. Re-evaluate tuned models
        print("Evaluating tuned models...")
        tuned_models = {"xgboost_tuned": xgb_tuned, "lightgbm_tuned": lgbm_tuned}
        df_tuned_metrics = evaluate_all_models(tuned_models, X_test, y_test)
        
        # 8. Generate SHAP plots for XGBoost (tuned) and LightGBM (tuned)
        generate_shap_summary(xgb_tuned, X_test, "xgboost_tuned")
        generate_shap_summary(lgbm_tuned, X_test, "lightgbm_tuned")
        explain_single_prediction(xgb_tuned, X_test, idx=0, model_name="xgboost_tuned")
        
        # 9. Build ensemble from tuned models
        best_ensemble = build_ensemble(xgb_tuned, lgbm_tuned, X_test, y_test)
        
        # 10. Print final summary
        best_single_name = df_tuned_metrics.iloc[0]["model"]
        best_single_val = df_tuned_metrics.iloc[0]["pr_auc"]
        print(f"Best single model: {best_single_name} with PR-AUC={best_single_val:.4f}. "
              f"Ensemble PR-AUC={best_ensemble['pr_auc']:.4f}.")

if __name__ == "__main__":
    main()
