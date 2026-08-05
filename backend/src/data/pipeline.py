"""Data engineering pipeline orchestrator."""

import os
import pickle
import pandas as pd
from src.data.validate import validate_raw_data, check_missing_values, check_class_imbalance
from src.data.preprocess import engineer_features, scale_features

# Constants
RAW_PATH: str = "data/raw/creditcard.csv"
PROCESSED_PATH: str = "data/processed/creditcard_processed.csv"
SCALER_PATH: str = "data/processed/scaler.pkl"

def run_pipeline() -> None:
    """Runs the end-to-end data engineering pipeline.

    Validates the raw data, applies feature engineering, scales key features,
    and saves the processed dataset.
    """
    print("Starting Phase 1 Data Pipeline...")
    
    # Check if raw file exists
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at '{RAW_PATH}'. "
            f"Please download the Kaggle Credit Card Fraud dataset and place it there."
        )
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    
    # Load Raw Data
    print(f"Reading raw data from {RAW_PATH}...")
    df_raw = pd.read_csv(RAW_PATH)
    
    # Data Validation
    print("Performing data validation...")
    df_validated = validate_raw_data(df_raw)
    check_missing_values(df_validated)
    check_class_imbalance(df_validated)
    
    # Feature Engineering
    print("Performing feature engineering...")
    df_engineered = engineer_features(df_validated)
    
    # Scaling
    print("Scaling engineered features...")
    df_processed, scaler = scale_features(df_engineered)
    
    # Save Processed Data and Scaler
    print(f"Saving processed data to {PROCESSED_PATH}...")
    df_processed.to_csv(PROCESSED_PATH, index=False)
    
    print(f"Saving scaler to {SCALER_PATH}...")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    print("Phase 1 Data Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
