"""Vector store building, loading, and querying using ChromaDB."""

import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.llm.config import CHROMA_DIR, TOP_K, get_embeddings

def build_vector_store(chunks: List[Document]) -> Chroma:
    """Creates a Chroma vector store from Document chunks and persists it.

    Args:
        chunks: List of chunked Document objects.

    Returns:
        Chroma: The initialized Chroma vector store instance.
    """
    print(f"Building vector store with {len(chunks)} chunks...")
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Built vector store successfully. Saved to {CHROMA_DIR}")
    return vector_store

def load_vector_store() -> Chroma:
    """Loads an existing Chroma vector store from the local directory.

    Returns:
        Chroma: The loaded Chroma vector store instance.
    """
    print(f"Loading vector store from {CHROMA_DIR}...")
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vector_store

def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K
) -> List[Document]:
    """Runs a similarity search on the vector store for a given query.

    Args:
        vector_store: Chroma vector store instance.
        query: Query string.
        k: Number of top documents to retrieve.

    Returns:
        List[Document]: Top-k retrieved Document objects.
    """
    print(f"Top {k} chunks retrieved for query: '{query[:60]}...'")
    return vector_store.similarity_search(query, k=k)
