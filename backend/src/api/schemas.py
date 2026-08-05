"""Pydantic schemas for the FinShield FastAPI endpoints."""

from __future__ import annotations

import uuid
from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class TransactionInput(BaseModel):
    V1: float = Field(0.0, description="PCA component V1")
    V2: float = Field(0.0, description="PCA component V2")
    V3: float = Field(0.0, description="PCA component V3")
    V4: float = Field(0.0, description="PCA component V4")
    V5: float = Field(0.0, description="PCA component V5")
    V6: float = Field(0.0, description="PCA component V6")
    V7: float = Field(0.0, description="PCA component V7")
    V8: float = Field(0.0, description="PCA component V8")
    V9: float = Field(0.0, description="PCA component V9")
    V10: float = Field(0.0, description="PCA component V10")
    V11: float = Field(0.0, description="PCA component V11")
    V12: float = Field(0.0, description="PCA component V12")
    V13: float = Field(0.0, description="PCA component V13")
    V14: float = Field(0.0, description="PCA component V14")
    V15: float = Field(0.0, description="PCA component V15")
    V16: float = Field(0.0, description="PCA component V16")
    V17: float = Field(0.0, description="PCA component V17")
    V18: float = Field(0.0, description="PCA component V18")
    V19: float = Field(0.0, description="PCA component V19")
    V20: float = Field(0.0, description="PCA component V20")
    V21: float = Field(0.0, description="PCA component V21")
    V22: float = Field(0.0, description="PCA component V22")
    V23: float = Field(0.0, description="PCA component V23")
    V24: float = Field(0.0, description="PCA component V24")
    V25: float = Field(0.0, description="PCA component V25")
    V26: float = Field(0.0, description="PCA component V26")
    V27: float = Field(0.0, description="PCA component V27")
    V28: float = Field(0.0, description="PCA component V28")
    time_hour: float = Field(..., ge=0.0, lt=24.0, description="Hour of transaction")
    amount: float = Field(..., gt=0.0, description="Transaction amount in euros")
    is_night: int = Field(0, ge=0, le=1)
    is_weekend: int = Field(0, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.3598,
                "V2": -0.0727,
                "V3": 2.5363,
                "V4": 1.3781,
                "V5": -0.3383,
                "V6": 0.4623,
                "V7": 0.2396,
                "V8": 0.0987,
                "V9": 0.3638,
                "V10": 0.0908,
                "V11": -0.5516,
                "V12": -0.6178,
                "V13": -0.9914,
                "V14": -0.3111,
                "V15": 1.4682,
                "V16": -0.4704,
                "V17": 0.2080,
                "V18": 0.0258,
                "V19": 0.4040,
                "V20": 0.2514,
                "V21": -0.0183,
                "V22": 0.2778,
                "V23": -0.1105,
                "V24": 0.0669,
                "V25": 0.1285,
                "V26": -0.1891,
                "V27": 0.1336,
                "V28": -0.0211,
                "time_hour": 2.3,
                "amount": 847.23,
                "is_night": 1,
                "is_weekend": 0
            }
        }
    }

class FraudPredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    is_fraud: bool
    top_features: List[Dict[str, Any]]
    explanation: str
    policy_reference: str
    processing_time_ms: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_loaded: bool
    uptime_seconds: float

class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]

    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v: List[TransactionInput]) -> List[TransactionInput]:
        if len(v) > 100:
            raise ValueError("Batch size exceeds limit of 100 transactions")
        return v

class ChatRequest(BaseModel):
    query: str = Field(..., description="The policy or compliance question to ask")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The grounded RAG generated answer")
    sources: List[str] = Field(default=[], description="Source policy files cited")
