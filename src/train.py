"""
Model training script for the credit risk model.

Supports:
- Logistic Regression with WoE-encoded features (interpretable, Basel II friendly)
- Gradient Boosting (XGBoost / LightGBM) for comparison

Usage:
    python -m src.train --data data/raw/xente.csv --model lr
    python -m src.train --data data/raw/xente.csv --model xgb
"""

import argparse
import joblib
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

from src.data_processing import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def get_model(model_type: str, random_state: int = 42):
    """Return an unfitted estimator based on model_type."""
    if model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'lr' or 'xgb'.")


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute and log evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["f1_score"] = report["weighted avg"]["f1-score"]

    log.info("ROC-AUC: %.4f", metrics["roc_auc"])
    log.info("Avg Precision: %.4f", metrics["avg_precision"])
    log.info("Weighted F1: %.4f", metrics["f1_score"])
    return metrics


def train(data_path: str, model_type: str = "lr", random_state: int = 42) -> None:
    """Full training run with MLflow tracking."""
    log.info("Loading and preprocessing data from %s", data_path)
    pipeline = run_pipeline(data_path)
    X_train = pipeline["X_train"]
    X_test = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_test = pipeline["y_test"]
    scaler = pipeline["scaler"]

    model = get_model(model_type, random_state)

    with mlflow.start_run(run_name=f"credit_risk_{model_type}"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        log.info("Training %s model...", model_type)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Persist artifacts
        model_path = MODEL_DIR / f"{model_type}_model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        log.info("Model saved to %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk model")
    parser.add_argument("--data", required=True, help="Path to raw CSV data")
    parser.add_argument("--model", default="lr", choices=["lr", "xgb"], help="Model type")
    parser.add_argument("--seed", type=int, default=42, help="Random state")
    args = parser.parse_args()
    train(args.data, args.model, args.seed)
