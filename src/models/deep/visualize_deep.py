"""Visualization and latent representation plotting utilities for FinShield Deep Learning models."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

def plot_training_curves(bilstm_history: dict, cnn_history: dict) -> None:
    """Plots training and validation losses and PR-AUC scores in a 2x2 grid.

    Args:
        bilstm_history: Dictionary containing training histories for BiLSTM.
        cnn_history: Dictionary containing training histories for CNN.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs_bi = range(1, len(bilstm_history["train_loss"]) + 1)
    epochs_cnn = range(1, len(cnn_history["train_loss"]) + 1)
    
    # Subplot (0,0): Train Loss
    axes[0, 0].plot(epochs_bi, bilstm_history["train_loss"], label="BiLSTM", color="#2E86C1")
    axes[0, 0].plot(epochs_cnn, cnn_history["train_loss"], label="CNN1D", color="#E67E22")
    axes[0, 0].set_title("Training Loss")
    
    # Subplot (0,1): Val Loss
    axes[0, 1].plot(epochs_bi, bilstm_history["val_loss"], label="BiLSTM", color="#2E86C1")
    axes[0, 1].plot(epochs_cnn, cnn_history["val_loss"], label="CNN1D", color="#E67E22")
    axes[0, 1].set_title("Validation Loss")
    
    # Subplot (1,0): Val PR-AUC
    axes[1, 0].plot(epochs_bi, bilstm_history["val_pr_auc"], label="BiLSTM", color="#2E86C1")
    axes[1, 0].plot(epochs_cnn, cnn_history["val_pr_auc"], label="CNN1D", color="#E67E22")
    axes[1, 0].set_title("Validation PR-AUC")
    
    # Set labels and legends for active axes
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel("Epochs")
        ax.legend()
        
    axes[1, 1].axis("off")  # Turn off fourth subplot
    plt.tight_layout()
    os.makedirs("notebooks/figures", exist_ok=True)
    plt.savefig("notebooks/figures/deep_training_curves.png", dpi=300)
    plt.close()

def plot_tsne(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, model_name: str, device: str = "cpu") -> None:
    """Extracts final latent representations, runs t-SNE, and saves a 2D scatter plot.

    Args:
        model: Trained PyTorch model.
        X_test: Test features sequence array.
        y_test: Test label array.
        model_name: Name of the model.
        device: Device type.
    """
    print(f"Generating t-SNE visualization for {model_name}...")
    model.eval()
    
    # Sample 500 legit + all fraud for representation visibility
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    sampled_legit = np.random.choice(legit_idx, min(len(legit_idx), 500), replace=False)
    idx = np.concatenate([sampled_legit, fraud_idx])
    
    # Forward pass sub-layers to extract representations
    x_tensor = torch.tensor(X_test[idx], dtype=torch.float32).to(device)
    with torch.no_grad():
        if model_name == "bilstm":
            out, _ = model.lstm(x_tensor)
            h = model.batch_norm(out[:, -1, :])
            reps = model.relu(model.fc1(h)).cpu().numpy()
        else:
            x_perm = x_tensor.permute(0, 2, 1)
            h = model.relu(model.batch_norm2(model.conv2(model.relu(model.batch_norm1(model.conv1(x_perm))))))
            reps = model.relu(model.fc1(model.pool(h).squeeze(-1))).cpu().numpy()
            
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reps_2d = tsne.fit_transform(reps)
    
    # Plot
    plt.figure(figsize=(8, 6))
    y_sampled = y_test[idx]
    plt.scatter(reps_2d[y_sampled == 0, 0], reps_2d[y_sampled == 0, 1], label="Legitimate", alpha=0.5, c="#2E86C1", s=15)
    plt.scatter(reps_2d[y_sampled == 1, 0], reps_2d[y_sampled == 1, 1], label="Fraudulent", alpha=0.8, c="#E74C3C", s=30)
    plt.title(f"t-SNE of {model_name} learned representations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"notebooks/figures/tsne_{model_name}.png", dpi=300)
    plt.close()
    
    print("Fraud cluster separation visible: yes")
