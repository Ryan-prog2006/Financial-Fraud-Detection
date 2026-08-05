"""SQLite database creation and natural language SQL agent for transaction querying."""

import os
import sqlite3
from typing import Any
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from src.llm.config import get_llm

def create_transaction_db(
    processed_path: str,
    db_path: str = "data/transactions.db"
) -> str:
    """Loads processed CSV and saves a subset of columns into a SQLite database.

    Args:
        processed_path: Path to the processed CSV file.
        db_path: Target path for the SQLite database.

    Returns:
        str: The path to the created SQLite database.
    """
    print(f"Creating SQLite database at {db_path}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Load and subset columns
    df = pd.read_csv(processed_path)
    cols = [
        "Hour", "Amount", "Is_night", "Is_weekend",
        "Amount_zscore", "Is_small_amount", "Is_round_amount", "Class"
    ]
    df_subset = df[cols].copy()
    df_subset.rename(columns={"Class": "is_fraud"}, inplace=True)
    
    # Write to SQLite
    conn = sqlite3.connect(db_path)
    df_subset.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"Created SQLite database with {len(df_subset)} transactions at {db_path}")
    return db_path

def build_sql_agent(db_path: str) -> Any:
    """Creates a LangChain SQL Database and builds a SQL agent.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Any: The initialized SQL agent executor.
    """
    print(f"Building SQL agent for database {db_path}...")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = get_llm("qwen/qwen3-32b")
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

def query_transactions(agent: Any, question: str) -> str:
    """Executes a natural language query against the transactions database.

    Args:
        agent: The SQL agent executor.
        question: Natural language question.

    Returns:
        str: Response text from the SQL agent.
    """
    print(f"\nQuerying SQL Agent: '{question}'")
    try:
        response = agent.invoke({"input": question})
        result = response.get("output", "").strip()
        print(f"Result: {result}")
        return result
    except Exception as e:
        err_msg = f"SQL Agent failed to process query due to error: {str(e)}"
        print(err_msg)
        return err_msg
