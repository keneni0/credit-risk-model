"""
Pydantic request/response models for the credit risk scoring API.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Single transaction input for risk scoring."""

    TransactionId: str = Field(..., description="Unique transaction identifier")
    BatchId: str = Field(..., description="Batch identifier")
    AccountId: str = Field(..., description="Account identifier")
    SubscriptionId: str = Field(..., description="Subscription identifier")
    CustomerId: str = Field(..., description="Customer identifier")
    CurrencyCode: str = Field(..., description="ISO currency code, e.g. UGX")
    CountryCode: int = Field(..., description="Country code")
    ProviderId: str = Field(..., description="Provider identifier")
    ProductId: str = Field(..., description="Product identifier")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Channel identifier")
    Amount: float = Field(..., description="Transaction amount (can be negative for debits)")
    Value: float = Field(..., ge=0, description="Absolute transaction value")
    TransactionStartTime: datetime = Field(..., description="Transaction timestamp (ISO 8601)")
    PricingStrategy: int = Field(..., ge=0, description="Pricing strategy code")

    @field_validator("Amount")
    @classmethod
    def amount_must_be_nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError("Amount must not be zero")
        return v


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for bulk scoring."""

    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)
    model_type: Literal["lr", "xgb"] = Field(default="lr", description="Model to use for scoring")


class RiskScoreResponse(BaseModel):
    """Risk score for a single transaction."""

    TransactionId: str
    CustomerId: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Probability of high credit risk")
    risk_label: int = Field(..., description="Binary risk label (1 = high risk, 0 = low risk)")
    model_type: str


class BatchRiskScoreResponse(BaseModel):
    """Batch of risk scores."""

    results: list[RiskScoreResponse]
    total: int
    high_risk_count: int


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    version: str = "0.1.0"
