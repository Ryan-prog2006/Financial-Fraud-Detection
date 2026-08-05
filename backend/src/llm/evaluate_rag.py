"""RAGAS-based evaluation suite for fraud policy Q&A."""

import sys
import types
try:
    from langchain_google_vertexai import ChatVertexAI
    mod = types.ModuleType('langchain_community.chat_models.vertexai')
    mod.ChatVertexAI = ChatVertexAI
    sys.modules['langchain_community.chat_models.vertexai'] = mod
except ImportError:
    pass

import os
import json
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from src.llm.config import get_llm, get_embeddings

def build_evaluation_dataset() -> List[Dict[str, str]]:
    """Creates a list of 5 question-answer pairs for RAGAS evaluation.

    Returns:
        List[Dict[str, str]]: Evaluation dataset with question and ground_truth.
    """
    return [
        {
            "question": "What is the RBI requirement for reporting fraud to authorities?",
            "ground_truth": "Banks must report fraud to RBI within 3 weeks of detection as per Master Directions on Fraud."
        },
        {
            "question": "What is a card-testing attack?",
            "ground_truth": "A card-testing attack involves making small test transactions to verify a stolen card works before making larger fraudulent purchases."
        },
        {
            "question": "What velocity threshold triggers a PCI DSS alert?",
            "ground_truth": "PCI DSS requires flagging if the same card is used more than 3 times within 10 minutes."
        },
        {
            "question": "What fraud pattern involves round number amounts?",
            "ground_truth": "Fraudsters often use exact round amounts when testing stolen cards or making fraudulent purchases."
        },
        {
            "question": "What is the recommended fraud score threshold for blocking a transaction?",
            "ground_truth": "Use threshold 0.5 for balanced precision and recall, or 0.3 when high recall is more important than precision."
        }
    ]

def run_ragas_evaluation(
    chain: Any,
    eval_dataset: List[Dict[str, str]]
) -> Dict[str, float]:
    """Runs RAGAS evaluation over the dataset and saves the results.

    Args:
        chain: The RetrievalQA chain.
        eval_dataset: List of question and ground truth dicts.

    Returns:
        Dict[str, float]: Calculated metrics dictionary.
    """
    print("\nRunning RAGAS evaluation (this might take a few minutes)...")
    questions, answers, contexts, ground_truths = [], [], [], []
    for item in eval_dataset:
        res = chain.invoke({"query": item["question"]})
        questions.append(item["question"])
        answers.append(res.get("result", "").strip())
        contexts.append([doc.page_content for doc in res.get("source_documents", [])])
        ground_truths.append(item["ground_truth"])
        
    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_llm(),
        embeddings=get_embeddings()
    )
    
    # Save and return
    import math
    dict_results = {}
    
    scores_dict_list = []
    if hasattr(result.scores, "to_dict"):
        scores_dict_list = result.scores.to_dict(orient="records")
    elif isinstance(result.scores, list):
        scores_dict_list = result.scores
        
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        name = metric.name
        scores = []
        for row in scores_dict_list:
            if isinstance(row, dict) and name in row:
                val = row[name]
                if val is not None and not math.isnan(val):
                    scores.append(val)
        dict_results[name] = sum(scores) / len(scores) if scores else 0.0
        
    os.makedirs("data/models", exist_ok=True)
    with open("data/models/ragas_results.json", "w") as f:
        json.dump(dict_results, f, indent=4)
        
    print_evaluation_summary(dict_results)
    return dict_results

def print_evaluation_summary(result: Dict[str, float]) -> None:
    """Helper to display the RAGAS results in a clean table format.

    Args:
        result: Metrics dict.
    """
    avg_score = sum(result.values()) / len(result)
    print("\n=== RAGAS Evaluation Results ===")
    for metric, val in result.items():
        print(f"{metric.replace('_', ' ').title():<18}: {val:.3f}")
    print(f"{'Average Score':<18}: {avg_score:.3f}")
