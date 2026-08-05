"""Data preprocessing and feature engineering module."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Constants
NIGHT_START_HOUR: int = 0
NIGHT_END_HOUR: int = 6
SMALL_AMOUNT_THRESHOLD: float = 5.0
COLS_TO_SCALE: list[str] = ["Amount", "Amount_log", "Amount_zscore"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers transactional features from the raw Credit Card dataset.

    Args:
        df: Input DataFrame with V1-V28, Time, Amount, and optionally Class.

    Returns:
        DataFrame with new engineered features and the raw 'Time' column dropped.
    """
    df_feat = df.copy()
    
    # Hour and Day extraction
    hour = (df_feat["Time"] // 3600) % 24
    day = (df_feat["Time"] // 86400)
    
    # Feature additions
    df_feat["Hour"] = hour.astype(float)
    df_feat["Day"] = day.astype(float)
    df_feat["Is_night"] = ((hour >= NIGHT_START_HOUR) & (hour <= NIGHT_END_HOUR)).astype(int)
    df_feat["Is_weekend"] = (day % 7 >= 5).astype(int)
    df_feat["Amount_log"] = np.log1p(df_feat["Amount"])
    
    mean_amt = df_feat["Amount"].mean()
    std_amt = df_feat["Amount"].std()
    df_feat["Amount_zscore"] = (df_feat["Amount"] - mean_amt) / (std_amt + 1e-9)
    
    df_feat["Is_round_amount"] = (df_feat["Amount"] % 1.0 == 0).astype(int)
    df_feat["Is_small_amount"] = (df_feat["Amount"] < SMALL_AMOUNT_THRESHOLD).astype(int)
    
    # V features interactions
    df_feat["V14_V4_interaction"] = df_feat["V14"] * df_feat["V4"]
    df_feat["V14_V10_interaction"] = df_feat["V14"] * df_feat["V10"]
    df_feat["V14_squared"] = df_feat["V14"] ** 2
    df_feat["V10_squared"] = df_feat["V10"] ** 2
    
    # Drop Time
    df_feat = df_feat.drop(columns=["Time"])
    return df_feat

def scale_features(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Scales Amount, Amount_log, and Amount_zscore using a StandardScaler.

    Args:
        df: Input DataFrame with features engineered.
        scaler: Optional pre-fitted StandardScaler. If None, fits a new scaler.

    Returns:
        A tuple of (scaled DataFrame, fitted StandardScaler).
    """
    df_scaled = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df_scaled[COLS_TO_SCALE] = scaler.fit_transform(df_scaled[COLS_TO_SCALE])
    else:
        df_scaled[COLS_TO_SCALE] = scaler.transform(df_scaled[COLS_TO_SCALE])
        
    return df_scaled, scaler

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Splits the dataset into feature matrix X and target series y.

    Args:
        df: Input DataFrame containing the target column 'Class'.

    Returns:
        A tuple of (X, y) where X is the feature DataFrame and y is the Class series.
    """
    if "Class" not in df.columns:
        raise KeyError("Target column 'Class' not found in DataFrame.")
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()
    return X, y
