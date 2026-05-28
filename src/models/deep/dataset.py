"""Sequence dataset creation and dataloader utilities for FinShield Deep Learning models."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
SEQUENCE_LENGTH: int = 10
PROCESSED_PATH: str = "data/processed/creditcard_processed.csv"

def build_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Converts tabular transactions into sequential windows sorted by time (Hour).

    Args:
        df: Input DataFrame containing processed features and 'Class'.
        seq_len: The length of each transaction history sequence window.

    Returns:
        A tuple of (X, y) as numpy arrays.
    """
    df_sorted = df.sort_values(by="Hour", ascending=True).copy()
    y_full = df_sorted["Class"].values
    X_full = df_sorted.drop(columns=["Class"]).values
    
    # Fast strided sliding window calculation using numpy
    X = np.lib.stride_tricks.sliding_window_view(X_full, window_shape=seq_len, axis=0)
    # Transpose from (n_samples, n_features, seq_len) to (n_samples, seq_len, n_features)
    X = np.transpose(X, (0, 2, 1))
    y = y_full[seq_len - 1:]
    
    n_samples = len(X)
    fraud_count = int(y.sum())
    rate = (fraud_count / n_samples) * 100
    print(f"Built {n_samples} sequences. Fraud sequences: {fraud_count} ({rate:.3f}%)")
    
    return X, y

def split_sequences(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits sequence arrays into stratified train and test sets.

    Args:
        X: Sequence feature array.
        y: Sequence label array.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Class distribution - Train:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("Class distribution - Test:")
    print(pd.Series(y_test).value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test

class FraudSequenceDataset(Dataset):
    """PyTorch Dataset wrapper for sequence transaction data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initializes the dataset with tensors."""
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns length of dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets sequence features and target at index."""
        return self.X[idx], self.y[idx]

def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256
) -> tuple[DataLoader, DataLoader]:
    """Creates train and test dataloaders with weighted training sampler.

    Args:
        X_train: Train features.
        y_train: Train labels.
        X_test: Test features.
        y_test: Test labels.
        batch_size: DataLoader batch size.

    Returns:
        A tuple of (train_loader, test_loader).
    """
    train_dataset = FraudSequenceDataset(X_train, y_train)
    test_dataset = FraudSequenceDataset(X_test, y_test)
    
    fraud_count = int(y_train.sum())
    legit_count = len(y_train) - fraud_count
    
    weight_fraud = len(y_train) / (2.0 * fraud_count)
    weight_legit = len(y_train) / (2.0 * legit_count)
    
    sample_weights = np.where(y_train == 1, weight_fraud, weight_legit)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
