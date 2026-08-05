"""Feature importance formatting and natural language fraud explanation generation."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

def format_transaction_for_llm(
    transaction: Dict[str, Any],
    shap_values: Dict[str, float]
) -> str:
    """Formats transaction details and SHAP values for LLM consumption.

    Args:
        transaction: Dict mapping feature names to values.
        shap_values: Dict mapping feature names to SHAP values.

    Returns:
        str: Formatted transaction and SHAP details.
    """
    details = ["Transaction Details:"]
    amt = transaction.get("Amount", 0.0)
    amt_z = transaction.get("Amount_zscore", 0.0)
    details.append(f"Amount: €{amt:.2f} (z-score: {amt_z:+.2f})")
    
    hour = transaction.get("Hour", 0.0)
    is_night = int(transaction.get("Is_night", 0))
    details.append(f"Hour: {hour:.1f} (Is_night: {is_night})")
    
    for k in ["Is_weekend", "Is_small_amount", "Is_round_amount"]:
        if k in transaction:
            details.append(f"{k}: {int(transaction[k])}")
            
    details.append("\nTop features driving fraud prediction:")
    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in sorted_shap[:5]:
        annotation = ""
        if feat == "Amount_zscore":
            annotation = f" (amount is {amt_z:.1f} standard deviations from mean)"
        elif feat == "Is_night" and val > 0:
            annotation = " (night-time transaction increases fraud risk)"
        elif feat == "Hour" and val > 0:
            annotation = " (unusual hour increases fraud risk)"
        elif feat == "Is_round_amount" and val > 0:
            annotation = " (exact round amount matches cash-out pattern)"
        elif feat.startswith("V"):
            annotation = " (latent feature indicating high correlation with fraud)"
        details.append(f"{feat}: SHAP={val:+.2f}{annotation}")
        
    return "\n".join(details)

def explain_flagged_transaction(
    chain: Any,
    transaction: Dict[str, Any],
    shap_values: Dict[str, float],
    fraud_score: float
) -> str:
    """Explains a single flagged transaction by querying the RAG chain.

    Args:
        chain: The RetrievalQA chain or QA agent.
        transaction: Dict of features to values.
        shap_values: Dict of features to SHAP values.
        fraud_score: The probability of fraud from the ML model.

    Returns:
        str: LLM-generated explanation.
    """
    formatted_tx = format_transaction_for_llm(transaction, shap_values)
    prompt = (
        f"The following transaction has been flagged as potentially fraudulent "
        f"by our ML model with a fraud probability of {fraud_score:.1%}.\n\n"
        f"{formatted_tx}\n\n"
        f"Please provide:\n"
        f"1. A plain English explanation of why this transaction was flagged\n"
        f"2. Which specific fraud pattern from our documentation this matches\n"
        f"3. The relevant RBI or PCI DSS requirement that applies\n"
        f"4. Recommended action for the fraud analyst (block/review/escalate)"
    )
    res = chain.invoke({"query": prompt})
    explanation = res.get("result", "").strip()
    print(f"\n=== FRAUD EXPLANATION (Score: {fraud_score:.1%}) ===")
    print(explanation)
    print("==================================================")
    return explanation

def batch_explain_top_frauds(
    chain: Any,
    X_test: pd.DataFrame,
    y_scores: np.ndarray,
    shap_values: np.ndarray,
    top_n: int = 3
) -> List[str]:
    """Explains the top N transactions with the highest fraud scores.

    Args:
        chain: The RetrievalQA chain.
        X_test: Test features DataFrame.
        y_scores: Model prediction probabilities.
        shap_values: SHAP values array.
        top_n: Number of top frauds to explain.

    Returns:
        List[str]: List of explanation strings.
    """
    top_indices = np.argsort(y_scores)[-top_n:][::-1]
    explanations = []
    
    for idx in top_indices:
        row = X_test.iloc[idx].to_dict()
        score = float(y_scores[idx])
        shap_dict = dict(zip(X_test.columns, shap_values[idx]))
        explanation = explain_flagged_transaction(chain, row, shap_dict, score)
        explanations.append(explanation)
        
    return explanations
