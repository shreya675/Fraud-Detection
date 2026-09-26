"""Pydantic request / response models for the FraudGuard API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class Transaction(BaseModel):
    """One PaySim-style transaction as seen by a real-time scorer."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": 300,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }
    )

    step: int = Field(..., ge=0, description="Hour index in the simulation (1 step = 1 hour)")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., ge=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin balance before the transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Origin balance after the transaction")
    oldbalanceDest: float = Field(
        ..., ge=0, description="Destination balance before the transaction"
    )
    newbalanceDest: float = Field(
        ..., ge=0, description="Destination balance after the transaction"
    )

    @field_validator("type", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator(
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    )
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("must be a finite number")
        return v


class PredictRequest(BaseModel):
    transaction: Transaction
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override the trained decision threshold"
    )
    explain: bool = Field(True, description="Include SHAP top features and reason codes")
    top_n: int = Field(5, ge=1, le=26, description="Number of SHAP features to return")


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: Literal["increases_fraud_risk", "decreases_fraud_risk"]


class ExplanationOut(BaseModel):
    top_features: list[FeatureContribution]
    reasons: list[str]
    base_value: float


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_level: str
    explanation: ExplanationOut | None = None
    model_type: str
    prediction_id: int | None = None


class BatchRowResult(BaseModel):
    row: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    reasons: list[str] = []


class BatchSummary(BaseModel):
    rows: int
    flagged: int
    flag_rate: float
    threshold: float
    batch_id: str
    processing_ms: float


class PredictBatchResponse(BaseModel):
    summary: BatchSummary
    results: list[BatchRowResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_type: str | None = None
    version: str | None = None
    database: Literal["ok", "unavailable"]


class ErrorResponse(BaseModel):
    error: str
    detail: str | list | dict | None = None
