"""Master orchestrator to run Phase 4 LLM + RAG pipeline end-to-end."""

import os
import pickle
from typing import Any
import numpy as np
import pandas as pd
import shap
from src.llm.config import CHROMA_DIR, DOCS_DIR
from src.llm.document_loader import (
    create_synthetic_policy_docs, load_documents, chunk_documents
)
from src.llm.vector_store import build_vector_store
from src.llm.rag_chain import build_rag_chain, ask_fraud_policy
from src.llm.fraud_explainer import explain_flagged_transaction
from src.llm.sql_agent import create_transaction_db, build_sql_agent, query_transactions
from src.llm.evaluate_rag import build_evaluation_dataset, run_ragas_evaluation
from src.models.train import load_and_split, PROCESSED_PATH

def run_rag_setup() -> tuple:
    """Sets up synthetic docs, loads, chunks, builds vector store and RAG chain.

    Returns:
        tuple: (chain, num_chunks)
    """
    create_synthetic_policy_docs()
    documents = load_documents()
    chunks = chunk_documents(documents)
    vector_store = build_vector_store(chunks)
    chain = build_rag_chain(vector_store)
    return chain, len(chunks)

def run_demo_queries(chain: Any) -> None:
    """Runs 3 demo questions through the RAG chain and prints first 2 lines of answers.

    Args:
        chain: The RetrievalQA chain.
    """
    questions = [
        "What should a fraud analyst do when a card-testing attack is detected?",
        "What are the RBI reporting requirements for UPI fraud?",
        "What fraud pattern does a transaction at 2 AM with a z-score of +3.2 match?"
    ]
    for i, q in enumerate(questions, 1):
        res = ask_fraud_policy(chain, q)
        answer = res["answer"]
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        first_two_lines = "\n".join(lines[:2])
        print(f"\nQ{i}: {q}")
        print(f"A{i} (Preview):\n{first_two_lines}")

def run_shap_explainer(chain: Any) -> None:
    """Loads XGBoost model, computes SHAP values for one prediction and explains it.

    Args:
        chain: The RetrievalQA chain.
    """
    print("\n--- Running SHAP Explainer ---")
    X_train, X_test, y_train, y_test = load_and_split("data/processed/creditcard_processed.csv")
    
    with open("data/models/xgboost_tuned.pkl", "rb") as f:
        model = pickle.load(f)
        
    y_scores = model.predict_proba(X_test)[:, 1]
    top_indices = np.argsort(y_scores)[-1:][::-1]
    top_idx = top_indices[0]
    row = X_test.iloc[top_idx].to_dict()
    score = float(y_scores[top_idx])
    
    explainer = shap.TreeExplainer(model)
    shap_vals_top = explainer.shap_values(X_test.iloc[top_indices])
    if isinstance(shap_vals_top, list):
        shap_vals_top = shap_vals_top[1]
        
    shap_dict = dict(zip(X_test.columns, shap_vals_top[0]))
    explain_flagged_transaction(chain, row, shap_dict, score)

def run_sql_agent_queries() -> None:
    """Dumps transaction data to SQLite, builds the agent and runs 2 queries."""
    print("\n--- Running SQL Agent ---")
    db_path = create_transaction_db("data/processed/creditcard_processed.csv")
    agent = build_sql_agent(db_path)
    
    query_transactions(agent, "How many fraud transactions occurred at night?")
    query_transactions(agent, "What is the average amount of fraudulent transactions?")

def main() -> None:
    """Main orchestrator for running the Phase 4 pipeline end-to-end."""
    print("=== Starting Phase 4: LLM + RAG Intelligence Layer ===")
    
    chain, num_chunks = run_rag_setup()
    run_demo_queries(chain)
    run_shap_explainer(chain)
    run_sql_agent_queries()
    
    eval_ds = build_evaluation_dataset()
    eval_results = run_ragas_evaluation(chain, eval_ds)
    avg_score = sum(eval_results.values()) / len(eval_results)
    
    print("\n=== Phase 4 Summary ===")
    print(f"Documents indexed: {num_chunks} chunks in ChromaDB")
    print("RAG chain: operational")
    print("SQL agent: operational")
    print(f"RAGAS Average Score: {avg_score:.3f}")
    print("LLM Fraud Explainer: operational")

if __name__ == "__main__":
    main()
