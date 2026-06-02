"""
FastAPI application for the credit risk scoring service.

Endpoints:
    GET  /health          - Health check
    POST /score           - Score a single transaction
    POST /score/batch     - Score a batch of transactions
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.api.pydantic_models import (
    BatchRiskScoreResponse,
    BatchTransactionRequest,
    HealthResponse,
    RiskScoreResponse,
    TransactionRequest,
)
from src.predict import predict_proba, load_artifacts

log = logging.getLogger("uvicorn.error")

# Track loaded model state
_model_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the default model on startup."""
    try:
        model, scaler = load_artifacts("lr")
        _model_cache["lr"] = (model, scaler)
        log.info("Default LR model loaded successfully.")
    except FileNotFoundError:
        log.warning("No pre-trained model found. Train a model before scoring.")
    yield
    _model_cache.clear()


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Scores transaction records for credit default risk using trained ML models.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check() -> HealthResponse:
    """Return API health and model availability."""
    return HealthResponse(
        status="ok",
        model_loaded=bool(_model_cache),
    )


def _transaction_to_df(tx: TransactionRequest) -> pd.DataFrame:
    """Convert a Pydantic model to a single-row DataFrame."""
    return pd.DataFrame([tx.model_dump()])


@app.post("/score", response_model=RiskScoreResponse, tags=["Scoring"])
def score_transaction(request: TransactionRequest, model_type: str = "lr") -> RiskScoreResponse:
    """
    Score a single transaction for credit risk.

    - **model_type**: 'lr' (Logistic Regression) or 'xgb' (XGBoost)
    """
    df = _transaction_to_df(request)
    try:
        scores = predict_proba(df, model_type=model_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    score = float(scores[0])
    return RiskScoreResponse(
        TransactionId=request.TransactionId,
        CustomerId=request.CustomerId,
        risk_score=score,
        risk_label=int(score >= 0.5),
        model_type=model_type,
    )


@app.post("/score/batch", response_model=BatchRiskScoreResponse, tags=["Scoring"])
def score_batch(request: BatchTransactionRequest) -> BatchRiskScoreResponse:
    """
    Score a batch of transactions (up to 1000 per request).
    """
    df = pd.DataFrame([tx.model_dump() for tx in request.transactions])
    try:
        scores = predict_proba(df, model_type=request.model_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        log.exception("Batch prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    results = [
        RiskScoreResponse(
            TransactionId=tx.TransactionId,
            CustomerId=tx.CustomerId,
            risk_score=float(s),
            risk_label=int(s >= 0.5),
            model_type=request.model_type,
        )
        for tx, s in zip(request.transactions, scores)
    ]

    return BatchRiskScoreResponse(
        results=results,
        total=len(results),
        high_risk_count=sum(r.risk_label for r in results),
    )
