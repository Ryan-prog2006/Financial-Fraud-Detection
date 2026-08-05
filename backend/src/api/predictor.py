"""Singleton FraudPredictor class and inference functions for FinShield."""

from __future__ import annotations

import os
import time
import uuid
import pickle
import numpy as np
import pandas as pd
import shap
from typing import Any, Dict, List
from src.api.schemas import TransactionInput

MODEL_PATH = "data/models/xgboost_tuned.pkl"
SCALER_PATH = "data/processed/scaler.pkl"

from functools import lru_cache

_predictor_instance = None

@lru_cache(maxsize=1)
def get_cached_rag_chain():
    from src.llm.vector_store import load_vector_store
    from src.llm.rag_chain import build_rag_chain
    vs = load_vector_store()
    return build_rag_chain(vs)

class FraudPredictor:
    def __init__(self) -> None:
        self.start_time = time.time()
        
        # Load XGBoost tuned model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as fh:
            self.model = pickle.load(fh)
            
        # Load scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        with open(SCALER_PATH, "rb") as fh:
            self.scaler = pickle.load(fh)
            
        # Feature names in correct order from training
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            self.feature_names = [
                'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
                'Hour', 'Day', 'Is_night', 'Is_weekend', 'Amount_log', 'Amount_zscore',
                'Is_round_amount', 'Is_small_amount', 'V14_V4_interaction',
                'V14_V10_interaction', 'V14_squared', 'V10_squared'
            ]
            
        # Explainer
        self.explainer = shap.TreeExplainer(self.model)

def get_predictor() -> FraudPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FraudPredictor()
    return _predictor_instance

def preprocess_input(transaction: TransactionInput) -> pd.DataFrame:
    predictor = get_predictor()
    
    amount = transaction.amount
    amount_log = np.log1p(amount)
    
    # Extract mean and scale parameter for zscore from the scaler
    # Since scaler was fit on [Amount, Amount_log, Amount_zscore]
    mean_amt = predictor.scaler.mean_[0]
    std_amt = predictor.scaler.scale_[0]
    amount_zscore = (amount - mean_amt) / (std_amt + 1e-9)
    
    # Scale Amount, Amount_log, Amount_zscore using the scaler
    scaled_vals = predictor.scaler.transform([[amount, amount_log, amount_zscore]])
    scaled_amount = scaled_vals[0][0]
    scaled_amount_log = scaled_vals[0][1]
    scaled_amount_zscore = scaled_vals[0][2]
    
    # Build feature dictionary
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = getattr(transaction, f"V{i}")
        
    features["Amount"] = scaled_amount
    features["Hour"] = float(transaction.time_hour)
    features["Day"] = 0.0
    features["Is_night"] = int(transaction.is_night)
    features["Is_weekend"] = int(transaction.is_weekend)
    features["Amount_log"] = scaled_amount_log
    features["Amount_zscore"] = scaled_amount_zscore
    features["Is_round_amount"] = int(amount % 1.0 == 0)
    features["Is_small_amount"] = int(amount < 5.0)
    features["V14_V4_interaction"] = features["V14"] * features["V4"]
    features["V14_V10_interaction"] = features["V14"] * features["V10"]
    features["V14_squared"] = features["V14"] ** 2
    features["V10_squared"] = features["V10"] ** 2
    
    df = pd.DataFrame([features])
    
    # Check if any required feature is missing
    for col in predictor.feature_names:
        if col not in df.columns:
            raise ValueError(f"Required feature missing from engineered payload: {col}")
            
    # Return properly ordered DataFrame
    return df[predictor.feature_names]

def predict_transaction(transaction: TransactionInput, use_llm: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    predictor = get_predictor()
    
    # Preprocess
    df_preprocessed = preprocess_input(transaction)
    
    # Predict
    prob = float(predictor.model.predict_proba(df_preprocessed)[0, 1])
    is_fraud = prob >= 0.5
    
    # SHAP
    shap_vals = predictor.explainer(df_preprocessed)
    shap_arr = shap_vals.values[0]
    
    # Sort absolute shap values to get top 5
    top_indices = np.argsort(np.abs(shap_arr))[::-1][:5]
    top_features = []
    for idx in top_indices:
        feat_name = df_preprocessed.columns[idx]
        val = float(shap_arr[idx])
        top_features.append({
            "feature": feat_name,
            "shap_value": round(val, 4),
            "direction": "positive" if val > 0 else "negative"
        })
        
    # Risk level mapping
    if prob < 0.3:
        risk_level = "LOW"
    elif prob < 0.5:
        risk_level = "MEDIUM"
    elif prob < 0.75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
        
    top_feat_1 = top_features[0]["feature"]
    top_feat_2 = top_features[1]["feature"]
    
    # LLM Explanation + Policy Reference
    explanation = ""
    policy_reference = ""
    
    if use_llm and prob > 0.3:
        # Try calling Groq
        try:
            from src.llm.config import get_llm
            from langchain_core.messages import HumanMessage
            llm = get_llm()
            prompt = (
                f"Explain in exactly two simple sentences why a financial transaction was flagged as suspicious "
                f"by our machine learning model. The transaction amount is EUR {transaction.amount:.2f} "
                f"and was made at hour {transaction.time_hour:.1f}. The primary risk factors are "
                f"{top_feat_1} (SHAP contribution: {top_features[0]['shap_value']:+.4f}) and "
                f"{top_feat_2} (SHAP contribution: {top_features[1]['shap_value']:+.4f}). "
                f"Fraud probability is {prob:.1%}. Keep it plain English, direct, and concise."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            explanation = response.content.strip()
        except Exception:
            explanation = f"Transaction flagged due to {top_feat_1} and {top_feat_2}. Fraud probability: {prob:.1%}."
            
        # Try getting policy reference via RAG
        try:
            from src.llm.rag_chain import ask_fraud_policy
            
            chain = get_cached_rag_chain()
            rag_query = f"What specific regulation or PCI DSS section applies to a transaction flagged for fraud with top features {top_feat_1} and {top_feat_2}?"
            rag_res = ask_fraud_policy(chain, rag_query)
            if rag_res.get("sources"):
                policy_reference = ", ".join(rag_res["sources"])
            else:
                policy_reference = "RBI Master Circular / PCI DSS Section 10"
        except Exception:
            policy_reference = "RBI Master Directions on Fraud Management"
    else:
        explanation = f"Transaction flagged due to {top_feat_1} and {top_feat_2}. Fraud probability: {prob:.1%}."
        policy_reference = "General Regulatory Policy Guidelines"
        
    latency_ms = (time.perf_counter() - t0) * 1000
    
    return {
        "transaction_id": str(uuid.uuid4()),
        "fraud_probability": round(prob, 4),
        "risk_level": risk_level,
        "is_fraud": is_fraud,
        "top_features": top_features,
        "explanation": explanation,
        "policy_reference": policy_reference,
        "processing_time_ms": round(latency_ms, 2),
        "model_version": "xgboost_tuned_v1"
    }
