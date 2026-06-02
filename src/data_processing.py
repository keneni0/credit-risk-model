"""
Data processing pipeline for the credit risk model.

Responsibilities:
- Load raw transaction data
- Engineer features (RFM aggregates, time-based, behavioral)
- Encode categorical variables (WoE or label encoding)
- Handle missing values and outliers
- Split into train/test sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    df = pd.read_csv(filepath, parse_dates=["TransactionStartTime"])
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from TransactionStartTime."""
    df = df.copy()
    df["tx_hour"] = df["TransactionStartTime"].dt.hour
    df["tx_day_of_week"] = df["TransactionStartTime"].dt.dayofweek
    df["tx_month"] = df["TransactionStartTime"].dt.month
    df["tx_year"] = df["TransactionStartTime"].dt.year
    return df


def compute_rfm_features(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) features per CustomerId.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions with CustomerId, TransactionStartTime, Amount.
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation. Defaults to max transaction date + 1 day.

    Returns
    -------
    pd.DataFrame
        Customer-level RFM features joined back to the transaction frame.
    """
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Amount", "sum"),
        )
        .reset_index()
    )
    df = df.merge(rfm, on="CustomerId", how="left")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categorical columns (placeholder; replace with WoE in production)."""
    if cat_cols is None:
        cat_cols = ["ProductCategory", "ChannelId", "PricingStrategy", "ProviderId"]
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Assemble the model-ready feature matrix X and target y.

    The proxy target (FraudResult) is used as a stand-in for default label
    when a direct credit-default label is unavailable.
    """
    drop_cols = [
        "TransactionId", "BatchId", "AccountId", "SubscriptionId",
        "CustomerId", "CurrencyCode", "CountryCode", "TransactionStartTime",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop + (["FraudResult"] if "FraudResult" in df.columns else []))
    y = df["FraudResult"] if "FraudResult" in df.columns else None
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def run_pipeline(filepath: str | Path) -> dict:
    """
    End-to-end preprocessing pipeline.

    Returns a dict with keys: X_train, X_test, y_train, y_test, scaler.
    """
    df = load_data(filepath)
    df = extract_time_features(df)
    df = compute_rfm_features(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    X, y = build_feature_matrix(df)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }
