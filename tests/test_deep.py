"""Unit tests for Phase 3 deep learning sequence models and datasets."""

import numpy as np
import pandas as pd
import torch
import pytest
from src.models.deep.dataset import build_sequences, FraudSequenceDataset
from src.models.deep.bilstm import BiLSTMFraudDetector
from src.models.deep.cnn1d import CNN1DFraudDetector

def create_dummy_processed_data(num_rows: int = 50) -> pd.DataFrame:
    """Creates a dummy processed transactions DataFrame for testing.

    Args:
        num_rows: Number of rows to generate.

    Returns:
        DataFrame.
    """
    np.random.seed(42)
    data = {
        "Hour": np.arange(num_rows, dtype=float),
        "Amount": np.random.uniform(0.0, 100.0, num_rows),
        "Class": np.random.choice([0, 1], size=num_rows, p=[0.9, 0.1])
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0.0, 1.0, num_rows)
    return pd.DataFrame(data)

def test_build_sequences_shape() -> None:
    """Test that built sequences have the correct 3D feature shape."""
    df = create_dummy_processed_data(50)
    X, y = build_sequences(df, seq_len=10)
    # 30 features = Hour, Amount, V1-V28
    assert X.shape == (41, 10, 30)
    assert y.shape == (41,)

def test_sequences_label_is_last_in_window() -> None:
    """Test that sequence labels match the Class of the last row in the window."""
    df = create_dummy_processed_data(50)
    df_sorted = df.sort_values(by="Hour", ascending=True).copy()
    expected_label = df_sorted["Class"].iloc[9]
    
    _, y = build_sequences(df, seq_len=10)
    assert y[0] == expected_label

def test_bilstm_output_shape() -> None:
    """Test that the BiLSTM outputs correct logits shape (batch_size, 1)."""
    model = BiLSTMFraudDetector(input_size=10, hidden_size=16, num_layers=1)
    x = torch.randn(32, 10, 10)
    output = model(x)
    assert output.shape == (32, 1)

def test_cnn1d_output_shape() -> None:
    """Test that the 1D CNN outputs correct logits shape (batch_size, 1)."""
    model = CNN1DFraudDetector(input_size=10)
    x = torch.randn(32, 10, 10)
    output = model(x)
    assert output.shape == (32, 1)

def test_dataset_dtypes() -> None:
    """Test that FraudSequenceDataset returns float32 tensors."""
    X_dummy = np.random.normal(size=(5, 10, 10))
    y_dummy = np.random.choice([0, 1], size=5)
    
    dataset = FraudSequenceDataset(X_dummy, y_dummy)
    x_tensor, y_tensor = dataset[0]
    
    assert x_tensor.dtype == torch.float32
    assert y_tensor.dtype == torch.float32
