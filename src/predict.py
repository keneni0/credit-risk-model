"""
Prediction module for the credit risk model.

Loads a trained model and scaler from disk and returns
probability scores for new transaction records.
"""

import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import (
    extract_time_features,
    compute_rfm_features,
    handle_missing_values,
    encode_categoricals,
    build_feature_matrix,
)

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def load_artifacts(model_type: str = "lr") -> tuple:
    """Load model and scaler from disk."""
    model_path = MODEL_DIR / f"{model_type}_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing steps used during training."""
    df = extract_time_features(df)
    df = compute_rfm_features(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    X, _ = build_feature_matrix(df)
    return X


def predict_proba(df: pd.DataFrame, model_type: str = "lr") -> np.ndarray:
    """
    Return credit risk probability scores for a DataFrame of transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction records (same schema as training data).
    model_type : str
        Which saved model to use ('lr' or 'xgb').

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with probability of positive class (high risk).
    """
    model, scaler = load_artifacts(model_type)
    X = preprocess_input(df)

    # Align columns to training schema
    train_cols = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else X.columns
    X = X.reindex(columns=train_cols, fill_value=0)

    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]
    return probas


def predict_batch(csv_path: str, model_type: str = "lr", output_path: str | None = None) -> pd.DataFrame:
    """
    Score a CSV file and optionally save results.

    Returns a DataFrame with original index + risk_score column.
    """
    df = pd.read_csv(csv_path, parse_dates=["TransactionStartTime"])
    scores = predict_proba(df, model_type)

    result = df[["TransactionId", "CustomerId"]].copy()
    result["risk_score"] = scores
    result["risk_label"] = (scores >= 0.5).astype(int)

    if output_path:
        result.to_csv(output_path, index=False)
        log.info("Predictions saved to %s", output_path)

    return result
