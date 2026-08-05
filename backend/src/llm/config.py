"""Configuration, model loaders, and constant tokens for LLM & RAG."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment configuration
load_dotenv()

# Constants
GROQ_MODEL: str = "llama-3.1-8b-instant"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
CHROMA_DIR: str = "data/vectorstore/"
DOCS_DIR: str = "data/policy_docs/"
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
TOP_K: int = 4

def get_llm(model_name: str = GROQ_MODEL) -> ChatGroq:
    """Instantiates and returns the ChatGroq model wrapper.

    Args:
        model_name: Name of the Groq model to use.

    Raises:
        EnvironmentError: If GROQ_API_KEY is missing or empty.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key or key.strip() == "" or "your_key_here" in key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Get a free key at groq.com and add it to .env"
        )
    return ChatGroq(groq_api_key=key, model=model_name, temperature=0.0)

def get_embeddings() -> HuggingFaceEmbeddings:
    """Instantiates and returns the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
