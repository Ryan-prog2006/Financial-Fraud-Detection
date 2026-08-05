"""RAG chain construction and querying for fraud policy Q&A."""

import os
from typing import Dict, List, Any
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from src.llm.config import get_llm, TOP_K

PROMPT_TEMPLATE = """You are a fraud detection expert and compliance officer at an Indian bank.
You have access to RBI guidelines, PCI DSS requirements, and fraud pattern 
documentation. Answer the analyst's question accurately and cite the specific 
document and section your answer comes from.

Context from policy documents:
{context}

Analyst question: {question}

Provide a clear, structured answer. If the context contains relevant policy 
references, cite them explicitly (e.g., "Per RBI Master Directions Section X...").
If the question cannot be answered from the context, say so clearly.

Answer:"""

def build_rag_chain(vector_store: Chroma) -> RetrievalQA:
    """Builds and returns a LangChain RetrievalQA chain.

    Args:
        vector_store: The Chroma vector store instance to retrieve from.

    Returns:
        RetrievalQA: The configured RetrievalQA chain.
    """
    print("Building RAG chain...")
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("RAG chain successfully operational.")
    return chain

def ask_fraud_policy(chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """Queries the RAG chain and returns the answer and source documents.

    Args:
        chain: The RetrievalQA chain instance.
        question: Analyst's natural language question.

    Returns:
        Dict[str, Any]: A dictionary containing 'answer' and 'sources' list.
    """
    print(f"\nQuerying RAG: '{question}'")
    # Using invoke since __call__ is deprecated in newer langchain versions
    response = chain.invoke({"query": question})
    answer = response.get("result", "").strip()
    
    sources = []
    source_docs = response.get("source_documents", [])
    for doc in source_docs:
        source_path = doc.metadata.get("source", "")
        source_name = os.path.basename(source_path)
        if source_name and source_name not in sources:
            sources.append(source_name)
            
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    return {"answer": answer, "sources": sources}
