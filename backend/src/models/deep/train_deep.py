"""Training pipeline functions for FinShield Deep Learning models using PyTorch."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import mlflow

# Constants
EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
PATIENCE: int = 5
MODELS_DIR: str = "data/models/"

def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Computes pos_weight for BCEWithLogitsLoss to handle class imbalance.

    Args:
        y_train: Training label array.

    Returns:
        pos_weight tensor.
    """
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    pos_weight = legit_count / (fraud_count + 1e-9)
    return torch.tensor([pos_weight], dtype=torch.float32)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Executes a single training epoch.

    Args:
        model: PyTorch model.
        loader: Train data loader.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: CPU or GPU device name.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)

def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    """Evaluates the model on test/validation data for one epoch.

    Args:
        model: PyTorch model.
        loader: Evaluation data loader.
        criterion: Loss function.
        device: CPU or GPU device name.

    Returns:
        A tuple of (average evaluation loss, PR-AUC).
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    # Apply sigmoid to logits to obtain probabilities
    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    pr_auc = average_precision_score(all_targets, probs)
    return avg_loss, pr_auc

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    device: str = "cpu"
) -> tuple[nn.Module, dict]:
    """Trains a PyTorch model with early stopping, scheduler, and MLflow logging.

    Args:
        model: PyTorch model to train.
        train_loader: Training DataLoader.
        test_loader: Validation DataLoader.
        model_name: Name of the model.
        device: Hardware device to train on.

    Returns:
        A tuple of (trained best model, history dictionary).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    
    y_train = train_loader.dataset.y.numpy()
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
    best_pr_auc, epochs_no_improve = -1.0, 0
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, pr = evaluate_epoch(model, test_loader, criterion, device)
        scheduler.step(pr)
        
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_pr_auc"].append(pr)
        
        mlflow.log_metric(f"{model_name}_train_loss", tl, step=epoch)
        mlflow.log_metric(f"{model_name}_val_loss", vl, step=epoch)
        mlflow.log_metric(f"{model_name}_val_pr_auc", pr, step=epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val PR-AUC: {pr:.4f} | LR: {lr:.6f}")
        
        if pr > best_pr_auc:
            best_pr_auc, epochs_no_improve = pr, 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{model_name}_best.pt")))
    return model, history
