"""Data validation schema and reporting utilities using Pandera."""

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Constants
FRAUD_RATE_MIN: float = 0.001
FRAUD_RATE_MAX: float = 0.05

# Define columns V1 through V28 dynamically
_v_columns = {f"V{i}": Column(float, nullable=False) for i in range(1, 29)}

# Define the schema for raw creditcard data
credit_card_schema = DataFrameSchema(
    columns={
        "Time": Column(float, Check.ge(0.0), nullable=False),
        "Amount": Column(float, Check.ge(0.0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
        **_v_columns
    },
    coerce=True,
    strict=True
)

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates the input creditcard dataframe schema using Pandera.

    Args:
        df: The raw credit card transaction DataFrame.

    Returns:
        The validated DataFrame.

    Raises:
        SchemaError: If the dataframe does not conform to the schema.
    """
    print("Validating raw transaction data schema...")
    return credit_card_schema.validate(df)

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Checks for missing values in the DataFrame and prints a detailed report.

    Args:
        df: The DataFrame to analyze.

    Returns:
        The input DataFrame unchanged.
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print("--- Missing Value Report ---")
    if total_missing == 0:
        print("No missing values found in any columns.")
    else:
        for col, count in missing_counts.items():
            if count > 0:
                print(f"Column '{col}': {count} missing values ({count / len(df) * 100:.4f}%)")
    print(f"Total missing values: {total_missing}")
    return df

def check_class_imbalance(df: pd.DataFrame) -> float:
    """Checks and prints the class imbalance (fraud rate) for the transaction data.

    Args:
        df: The transaction DataFrame containing target column 'Class'.

    Returns:
        The calculated fraud rate.
    """
    if "Class" not in df.columns:
        raise ValueError("DataFrame must contain a 'Class' column to compute fraud rate.")
    
    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("DataFrame cannot be empty.")
        
    fraud_count = int(df["Class"].sum())
    fraud_rate = fraud_count / total_rows
    print("--- Class Imbalance Report ---")
    print(f"Total Transactions: {total_rows}")
    print(f"Fraudulent Transactions: {fraud_count}")
    print(f"Fraud Rate: {fraud_rate:.6%}")

    if not (FRAUD_RATE_MIN <= fraud_rate <= FRAUD_RATE_MAX):
        print(f"WARNING: Fraud rate ({fraud_rate:.6%}) is outside the expected range of "
              f"[{FRAUD_RATE_MIN:.1%}, {FRAUD_RATE_MAX:.1%}]. Check data distribution.")
    else:
        print("Fraud rate is within the healthy expected range.")
        
    return fraud_rate
