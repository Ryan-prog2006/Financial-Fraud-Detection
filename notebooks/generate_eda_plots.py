"""Script to generate beautiful EDA plots for FinShield platform."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1E1E1E',
    'axes.facecolor': '#1E1E1E',
    'text.color': '#FFFFFF',
    'axes.labelcolor': '#FFFFFF',
    'xtick.color': '#FFFFFF',
    'ytick.color': '#FFFFFF',
    'grid.color': '#444444'
})

RAW_PATH = "data/raw/creditcard.csv"
FIGURES_DIR = "notebooks/figures"

def main() -> None:
    """Loads dataset, performs exploratory data analysis, and saves 6 plots."""
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}")
        
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating EDA plots...")
    df = pd.read_csv(RAW_PATH)
    
    # Plot 1: Class Imbalance
    plt.figure(figsize=(8, 5))
    counts = df["Class"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette=["#2E86C1", "#E74C3C"])
    plt.yscale("log")
    plt.title("Class Imbalance (Log Scale)", fontsize=14, color="white")
    plt.xlabel("Class (0: Legitimate, 1: Fraud)", fontsize=12)
    plt.ylabel("Number of Transactions (Log)", fontsize=12)
    plt.xticks([0, 1], ["Legitimate", "Fraudulent"])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_imbalance.png"), dpi=300)
    plt.close()
    
    # Plot 2: Amount Distribution (Original vs Log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Amount"], bins=50, kde=True, ax=axes[0], color="#2E86C1")
    axes[0].set_title("Distribution of Transaction Amount", fontsize=12)
    axes[0].set_xlabel("Amount (€)")
    axes[0].set_yscale("log")
    
    amount_log = np.log1p(df["Amount"])
    sns.histplot(amount_log, bins=50, kde=True, ax=axes[1], color="#2ECC71")
    axes[1].set_title("Distribution of Log(Amount + 1)", fontsize=12)
    axes[1].set_xlabel("Log(Amount)")
    
    plt.suptitle("Transaction Amount Distribution Analysis", fontsize=14, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "amount_distribution.png"), dpi=300)
    plt.close()

    # Plot 3: Time distribution (Transaction Volume by Hour)
    plt.figure(figsize=(10, 5))
    hours = (df["Time"] // 3600) % 24
    sns.histplot(hours, bins=24, color="#E67E22", kde=True)
    plt.title("Transaction Volume by Hour of Day", fontsize=14, color="white")
    plt.xlabel("Hour of Day (0-23)", fontsize=12)
    plt.ylabel("Transaction Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "transaction_velocity.png"), dpi=300)
    plt.close()

    # Plot 4: Correlation Heatmap with Target
    plt.figure(figsize=(12, 8))
    corr = df.corr()["Class"].drop("Class").sort_values()
    sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
    plt.title("Feature Correlation with Target (Class)", fontsize=14, color="white")
    plt.xlabel("Pearson Correlation Coefficient", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # Plot 5: Time vs Amount Scatter (Colored by Class)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Time", y="Amount", hue="Class", palette=["#2E86C1", "#E74C3C"], alpha=0.6, s=15)
    plt.title("Time vs Amount Scatter Plot (Colored by Class)", fontsize=14, color="white")
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Amount (€)", fontsize=12)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "time_vs_amount.png"), dpi=300)
    plt.close()

    # Plot 6: Fraud Rate by Hour of Day
    plt.figure(figsize=(10, 5))
    df_temp = df.copy()
    df_temp["Hour"] = hours
    fraud_by_hour = df_temp[df_temp["Class"] == 1].groupby("Hour").size()
    sns.barplot(x=fraud_by_hour.index, y=fraud_by_hour.values, color="#E74C3C")
    plt.title("Fraudulent Transactions Count by Hour of Day", fontsize=14, color="white")
    plt.xlabel("Hour of Day (0-23)", fontsize=12)
    plt.ylabel("Number of Fraud Cases", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "hourly_fraud_count.png"), dpi=300)
    plt.close()

    print("Successfully generated all 6 EDA plots in notebooks/figures/!")

if __name__ == "__main__":
    main()
