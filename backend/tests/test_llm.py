"""Unit tests for the LLM + RAG intelligence layer."""

import os
import sqlite3
from src.llm.config import get_embeddings, DOCS_DIR
from src.llm.document_loader import create_synthetic_policy_docs, load_documents, chunk_documents
from src.llm.vector_store import build_vector_store, similarity_search
from src.llm.fraud_explainer import format_transaction_for_llm
from src.llm.sql_agent import create_transaction_db

def test_config_loads() -> None:
    """Checks that get_embeddings loads correctly."""
    embeddings = get_embeddings()
    assert embeddings is not None

def test_document_creation() -> None:
    """Verifies that synthetic policy files are correctly created."""
    create_synthetic_policy_docs()
    expected_files = [
        "rbi_fraud_guidelines.txt",
        "pci_dss_requirements.txt",
        "fraud_patterns_guide.txt",
        "finshield_system_guide.txt"
    ]
    for filename in expected_files:
        path = os.path.join(DOCS_DIR, filename)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            assert len(content) > 100

def test_chunking_produces_chunks() -> None:
    """Verifies document loading and chunk splitting."""
    create_synthetic_policy_docs()
    docs = load_documents()
    chunks = chunk_documents(docs)
    assert len(chunks) > 10
    for chunk in chunks:
        assert hasattr(chunk, "page_content")
        assert len(chunk.page_content) > 0

def test_vector_store_builds() -> None:
    """Verifies vector store initialization and similarity search."""
    create_synthetic_policy_docs()
    docs = load_documents()
    chunks = chunk_documents(docs)
    vector_store = build_vector_store(chunks)
    assert vector_store is not None
    results = similarity_search(vector_store, "RBI guidelines reporting", k=4)
    assert len(results) == 4
    for doc in results:
        assert hasattr(doc, "page_content")

def test_format_transaction_for_llm() -> None:
    """Checks the transaction detail formatting function."""
    mock_tx = {
        "Amount": 100.0,
        "Amount_zscore": 1.5,
        "Hour": 3.0,
        "Is_night": 1
    }
    mock_shap = {
        "Amount_zscore": 0.5,
        "Is_night": 0.3,
        "Hour": 0.1
    }
    formatted = format_transaction_for_llm(mock_tx, mock_shap)
    assert "Amount" in formatted
    assert "SHAP" in formatted

def test_transaction_db_created() -> None:
    """Verifies SQLite database generation from processed CSV."""
    db_path = "data/test_transactions.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    db = create_transaction_db("data/processed/creditcard_processed.csv", db_path)
    assert os.path.exists(db)
    
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(transactions);")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    
    expected_cols = [
        "Hour", "Amount", "Is_night", "Is_weekend",
        "Amount_zscore", "Is_small_amount", "Is_round_amount", "is_fraud"
    ]
    for col in expected_cols:
        assert col in columns
    
    if os.path.exists(db_path):
        os.remove(db_path)
