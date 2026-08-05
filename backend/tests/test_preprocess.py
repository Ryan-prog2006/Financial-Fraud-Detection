"""Unit tests for preprocessing and feature engineering functions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from src.data.preprocess import engineer_features, scale_features, split_features_target

def create_synthetic_data(num_rows: int = 10) -> pd.DataFrame:
    """Helper function to create synthetic credit card dataframe for testing.

    Args:
        num_rows: Number of rows to generate.

    Returns:
        DataFrame structured like the raw creditcard dataset.
    """
    np.random.seed(42)
    data = {
        "Time": np.linspace(0, 100000, num_rows),
        "Amount": np.random.uniform(0.5, 500.0, num_rows),
        "Class": np.random.choice([0, 1], size=num_rows, p=[0.9, 0.1]),
    }
    # Add V1 through V28 columns
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0.0, 1.0, num_rows)
    return pd.DataFrame(data)

def test_engineer_features_adds_expected_columns() -> None:
    """Test that all engineered features are correctly added to the DataFrame."""
    df_raw = create_synthetic_data()
    df_engineered = engineer_features(df_raw)
    
    expected_cols = [
        "Hour", "Day", "Is_night", "Is_weekend", "Amount_log",
        "Amount_zscore", "Is_round_amount", "Is_small_amount",
        "V14_V4_interaction", "V14_V10_interaction", "V14_squared", "V10_squared"
    ]
    for col in expected_cols:
        assert col in df_engineered.columns, f"Missing engineered column: {col}"

def test_engineer_features_drops_time() -> None:
    """Test that the original Time column is dropped during feature engineering."""
    df_raw = create_synthetic_data()
    df_engineered = engineer_features(df_raw)
    assert "Time" not in df_engineered.columns

def test_engineer_features_no_nulls_introduced() -> None:
    """Test that no NaN values are introduced by feature engineering."""
    df_raw = create_synthetic_data()
    df_engineered = engineer_features(df_raw)
    assert df_engineered.isnull().sum().sum() == 0

def test_split_features_target() -> None:
    """Test that split_features_target splits into feature matrix X and series y."""
    df_raw = create_synthetic_data()
    df_engineered = engineer_features(df_raw)
    X, y = split_features_target(df_engineered)
    
    assert "Class" not in X.columns
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
