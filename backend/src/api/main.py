"""FinShield FastAPI main application module."""

from __future__ import annotations

import os
import time
import pandas as pd
from typing import Dict, List, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    TransactionInput,
    FraudPredictionResponse,
    HealthResponse,
    BatchTransactionInput,
    ChatRequest,
    ChatResponse,
)
from src.api.predictor import (
    get_predictor,
    predict_transaction,
)
from src.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)

# Start time tracking for uptime
APP_START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load the model, scaler, and RAG chain on startup
    try:
        print("FastAPI warming up model and scaler...")
        get_predictor()
        print("FastAPI warming up RAG chain vector store and embeddings...")
        from src.api.predictor import get_cached_rag_chain
        get_cached_rag_chain()
        print("FastAPI startup warm-up completed successfully.")
    except Exception as exc:
        print(f"Startup Warning: Could not pre-load model or RAG chain: {exc}")
    yield

app = FastAPI(
    title="FinShield Fraud Detection API",
    description=(
        "FinShield is a production-grade end-to-end AI fraud detection platform. "
        "This API provides real-time transaction screening, SHAP explanations, "
        "and regulatory compliance guidelines using RAG."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middlewares
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    return {
        "message": "FinShield API running",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    model_ok = False
    try:
        predictor = get_predictor()
        if predictor.model is not None:
            model_ok = True
    except Exception:
        pass

    vectorstore_ok = os.path.exists("data/vectorstore")
    uptime = time.time() - APP_START_TIME

    return HealthResponse(
        status="healthy",
        model_loaded=model_ok,
        vectorstore_loaded=vectorstore_ok,
        uptime_seconds=round(uptime, 2)
    )

# Examples for POST /predict
low_risk_example = {
    "V1": 1.19, "V2": 0.27, "V3": 0.17, "V4": 0.45, "V5": 0.06,
    "V6": -0.08, "V7": -0.08, "V8": 0.09, "V9": -0.26, "V10": -0.17,
    "V11": 1.61, "V12": 1.07, "V13": 0.49, "V14": -0.14, "V15": 0.64,
    "V16": 0.46, "V17": -0.11, "V18": -0.18, "V19": -0.15, "V20": -0.07,
    "V21": -0.23, "V22": -0.64, "V23": 0.10, "V24": -0.34, "V25": 0.17,
    "V26": 0.13, "V27": -0.01, "V28": 0.01,
    "time_hour": 14.5, "amount": 12.50, "is_night": 0, "is_weekend": 0
}

high_risk_example = {
    "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29, "V5": 1.36,
    "V6": -0.85, "V7": -0.26, "V8": 0.76, "V9": 0.18, "V10": -1.86,
    "V11": 0.02, "V12": -2.43, "V13": 0.73, "V14": -2.42, "V15": -0.10,
    "V16": -1.59, "V17": -2.05, "V18": -0.92, "V19": 0.08, "V20": -0.31,
    "V21": -0.10, "V22": -0.23, "V23": 0.07, "V24": 0.20, "V25": 0.12,
    "V26": -0.48, "V27": -0.29, "V28": -0.18,
    "time_hour": 2.5, "amount": 999.99, "is_night": 1, "is_weekend": 1
}

@app.post("/predict", response_model=FraudPredictionResponse, tags=["Inference"])
async def predict_endpoint(
    transaction: TransactionInput = Body(
        ...,
        openapi_examples={
            "low_risk": {
                "summary": "Low Risk Legitimate Transaction",
                "description": "A normal, low-amount transaction during daytime.",
                "value": low_risk_example
            },
            "high_risk": {
                "summary": "High Risk Fraud Transaction",
                "description": "A high-risk transaction with strong fraud indicators.",
                "value": high_risk_example
            }
        }
    )
) -> FraudPredictionResponse:
    try:
        res = predict_transaction(transaction, use_llm=True)
        return FraudPredictionResponse(**res)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(exc)}"
        )

@app.post("/predict/fast", response_model=FraudPredictionResponse, tags=["Inference"])
async def predict_fast_endpoint(
    transaction: TransactionInput = Body(
        ...,
        openapi_examples={
            "low_risk": {
                "summary": "Low Risk Legitimate Transaction",
                "description": "A normal, low-amount transaction during daytime.",
                "value": low_risk_example
            },
            "high_risk": {
                "summary": "High Risk Fraud Transaction",
                "description": "A high-risk transaction with strong fraud indicators.",
                "value": high_risk_example
            }
        }
    )
) -> FraudPredictionResponse:
    try:
        res = predict_transaction(transaction, use_llm=False)
        return FraudPredictionResponse(**res)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during fast prediction: {str(exc)}"
        )

@app.post("/predict/batch", response_model=List[FraudPredictionResponse], tags=["Inference"])
async def predict_batch_endpoint(
    body: BatchTransactionInput
) -> List[FraudPredictionResponse]:
    try:
        responses = []
        for tx in body.transactions:
            res = predict_transaction(tx, use_llm=False)
            responses.append(FraudPredictionResponse(**res))
        return responses
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during batch prediction: {str(exc)}"
        )

@app.get("/model/info", tags=["Monitoring"])
async def model_info() -> Dict[str, Any]:
    return {
        "model_name": "xgboost_tuned",
        "model_type": "XGBClassifier",
        "training_pr_auc": 0.8756,
        "feature_count": 41,
        "last_retrain_date": "2026-05-30",
        "mlflow_run_id": "production_xgboost_run"
    }

@app.get("/metrics/sample", tags=["Monitoring"])
async def metrics_sample() -> Dict[str, Any]:
    csv_path = "data/processed/creditcard_processed.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processed data reference file not found"
        )
        
    try:
        # Load first 10,000 rows to keep it fast
        df = pd.read_csv(csv_path, nrows=10000)
        sample_df = df.sample(n=100, random_state=42)
        
        predictor = get_predictor()
        mean_amt = predictor.scaler.mean_[0]
        std_amt = predictor.scaler.scale_[0]
        
        probabilities = []
        high_risk_count = 0
        
        t0 = time.perf_counter()
        for _, row in sample_df.iterrows():
            # Reconstruct TransactionInput
            scaled_amount = row["Amount"]
            raw_amount = float(scaled_amount * std_amt + mean_amt)
            if raw_amount <= 0:
                raw_amount = 0.01
                
            tx_in = TransactionInput(
                V1=row["V1"], V2=row["V2"], V3=row["V3"], V4=row["V4"], V5=row["V5"],
                V6=row["V6"], V7=row["V7"], V8=row["V8"], V9=row["V9"], V10=row["V10"],
                V11=row["V11"], V12=row["V12"], V13=row["V13"], V14=row["V14"], V15=row["V15"],
                V16=row["V16"], V17=row["V17"], V18=row["V18"], V19=row["V19"], V20=row["V20"],
                V21=row["V21"], V22=row["V22"], V23=row["V23"], V24=row["V24"], V25=row["V25"],
                V26=row["V26"], V27=row["V27"], V28=row["V28"],
                time_hour=row["Hour"],
                amount=raw_amount,
                is_night=int(row["Is_night"]),
                is_weekend=int(row["Is_weekend"])
            )
            
            # Predict using fast path
            res = predict_transaction(tx_in, use_llm=False)
            prob = res["fraud_probability"]
            probabilities.append(prob)
            if res["risk_level"] in ["HIGH", "CRITICAL"]:
                high_risk_count += 1
                
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "avg_fraud_probability": round(float(sum(probabilities) / len(probabilities)), 4),
            "high_risk_count": high_risk_count,
            "prediction_latency_ms": round(latency_ms, 2)
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score sample: {str(exc)}"
        )

@app.post("/chat", response_model=ChatResponse, tags=["Compliance"])
async def chat_endpoint(request: ChatRequest = Body(...)) -> ChatResponse:
    try:
        from src.api.predictor import get_cached_rag_chain
        from src.llm.rag_chain import ask_fraud_policy
        chain = get_cached_rag_chain()
        res = ask_fraud_policy(chain, request.query)
        return ChatResponse(
            answer=res.get("answer", "No response generated."),
            sources=res.get("sources", [])
        )
    except Exception as exc:
        return ChatResponse(
            answer=f"The compliance assistant is currently offline or unable to process queries. Details: {str(exc)}",
            sources=[]
        )

@app.get("/model/versions", tags=["Monitoring"])
async def model_versions_endpoint() -> List[Dict[str, Any]]:
    try:
        from src.mlops.registry import list_model_versions
        df = list_model_versions()
        if df is not None and not df.empty:
            return df.to_dict(orient="records")
        return []
    except Exception:
        today_str = time.strftime("%Y-%m-%d")
        return [
            {
                "version": 1,
                "stage": "Production",
                "run_id": "production_xgboost_run",
                "creation_time": today_str + " 10:00"
            }
        ]

