"""Model explainability using SHAP."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def generate_shap_summary(model, X_test: pd.DataFrame, model_name: str) -> None:
    """Generates SHAP summary bar plot and prints top 10 important features.

    Args:
        model: Trained tree-based model object.
        X_test: Test features.
        model_name: Name of the model.
    """
    print(f"Generating SHAP summary for {model_name}...")
    X_sample = X_test.head(100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Save summary bar plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title(f"SHAP Feature Importance Summary - {model_name}")
    plt.tight_layout()
    os.makedirs("notebooks/figures", exist_ok=True)
    plt.savefig(f"notebooks/figures/shap_{model_name}.png", dpi=300)
    plt.close()
    
    # Print top 10 important features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": X_sample.columns, "importance": mean_abs_shap})
    df_imp = df_imp.sort_values(by="importance", ascending=False).head(10)
    print(f"Top 10 features for {model_name}:")
    print(df_imp.to_string(index=False))

def explain_single_prediction(model, X_test: pd.DataFrame, idx: int, model_name: str) -> None:
    """Computes SHAP waterfall plot and explains a single transaction prediction.

    Args:
        model: Trained tree-based model object.
        X_test: Test features.
        idx: Index of transaction in X_test.
        model_name: Name of the model.
    """
    print(f"Explaining single prediction for {model_name} at index {idx}...")
    X_single = X_test.iloc[[idx]]
    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(X_single)
    
    # Plot waterfall
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_exp[0], show=False)
    plt.title(f"SHAP Prediction Explanation - {model_name} (Index {idx})")
    plt.tight_layout()
    os.makedirs("notebooks/figures", exist_ok=True)
    plt.savefig(f"notebooks/figures/shap_single_{model_name}_{idx}.png", dpi=300)
    plt.close()
    
    # Print text explanation
    score = model.predict_proba(X_single)[0, 1]
    values = shap_exp.values[0]
    contributions = sorted(zip(X_test.columns, values), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Transaction {idx} was scored {score:.3f} because:")
    for feature, val in contributions[:5]:
        direction = "UP" if val > 0 else "DOWN"
        print(f"  {feature} pushed score {direction} by {val:.4f}")
