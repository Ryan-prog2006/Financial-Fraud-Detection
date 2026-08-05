"""Master pipeline orchestrator for FinShield Phase 3 Deep Learning models."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow
import src.models.deep.train_deep as train_deep
from src.models.deep.dataset import (
    build_sequences,
    split_sequences,
    create_dataloaders,
    PROCESSED_PATH
)
from src.models.deep.bilstm import BiLSTMFraudDetector
from src.models.deep.cnn1d import CNN1DFraudDetector
from src.models.deep.evaluate_deep import (
    get_deep_predictions,
    compute_deep_metrics,
    compare_with_classical
)
from src.models.deep.visualize_deep import (
    plot_training_curves,
    plot_tsne
)

def load_and_prepare_data() -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Loads processed tabular transactions and converts them into time-series dataloaders.

    Returns:
        A tuple of (train_loader, test_loader, X_test, y_test).
    """
    df = pd.read_csv(PROCESSED_PATH)
    X, y = build_sequences(df)
    X_train, X_test, y_train, y_test = split_sequences(X, y)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    return train_loader, test_loader, X_test, y_test

def evaluate_and_plot(
    bilstm: torch.nn.Module,
    cnn1d: torch.nn.Module,
    test_loader: DataLoader,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    bi_hist: dict,
    cnn_hist: dict
) -> None:
    """Evaluates Deep Learning models, plots summaries, and prints comparative summaries."""
    print("Evaluating deep learning models...")
    y_true, y_scores_bi = get_deep_predictions(bilstm, test_loader, device)
    bi_metrics = compute_deep_metrics(y_true, y_scores_bi, "BiLSTM")
    
    _, y_scores_cnn = get_deep_predictions(cnn1d, test_loader, device)
    cnn_metrics = compute_deep_metrics(y_true, y_scores_cnn, "CNN1D")
    
    plot_training_curves(bi_hist, cnn_hist)
    plot_tsne(bilstm, X_test, y_test, "bilstm", device)
    plot_tsne(cnn1d, X_test, y_test, "cnn1d", device)
    
    compare_with_classical({"BiLSTM": bi_metrics, "CNN1D": cnn_metrics})
    improve = ((max(bi_metrics["pr_auc"], cnn_metrics["pr_auc"]) - 0.8756) / 0.8756) * 100
    
    print("\n=== Phase 3 Summary ===")
    print(f"BiLSTM  PR-AUC: {bi_metrics['pr_auc']:.4f} | F1: {bi_metrics['f1']:.4f}")
    print(f"CNN1D   PR-AUC: {cnn_metrics['pr_auc']:.4f} | F1: {cnn_metrics['f1']:.4f}")
    print("Classical Ensemble PR-AUC: 0.8756 | F1: 0.8705")
    print(f"Deep learning improvement over classical: {improve:+.2f}%")

def run_deep_pipeline() -> None:
    """Orchestrates Phase 3 training, evaluation, and visualizations."""
    print("Initializing Phase 3 Deep Learning Pipeline...")
    
    # Adjust epochs and patience for fast verification on CPU/GPU
    train_deep.EPOCHS = 15
    train_deep.PATIENCE = 3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using hardware device: {device}")
    
    mlflow.set_experiment("finshield_deep_learning")
    train_loader, test_loader, X_test, y_test = load_and_prepare_data()
    input_size = X_test.shape[2]
    
    # Train BiLSTM
    with mlflow.start_run(run_name="bilstm_pipeline"):
        bilstm = BiLSTMFraudDetector(input_size=input_size)
        bilstm, bi_hist = train_deep.train_model(bilstm, train_loader, test_loader, "bilstm", device)
        
    # Train CNN1D
    with mlflow.start_run(run_name="cnn1d_pipeline"):
        cnn1d = CNN1DFraudDetector(input_size=input_size)
        cnn1d, cnn_hist = train_deep.train_model(cnn1d, train_loader, test_loader, "cnn1d", device)
        
    evaluate_and_plot(bilstm, cnn1d, test_loader, X_test, y_test, device, bi_hist, cnn_hist)

if __name__ == "__main__":
    run_deep_pipeline()
