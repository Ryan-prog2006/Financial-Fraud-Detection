# FinShield Fraud Detection — Model Card

> Prepared in accordance with the Google Model Cards framework  
> (Mitchell et al., 2019, https://arxiv.org/abs/1810.03993)

---

## 1. Model Details

| Field | Value |
|-------|-------|
| **Model name** | FinShield Fraud Detector |
| **Version** | 1.0.0 |
| **Model type** | XGBoost Gradient Boosted Trees (binary classifier) |
| **Ensemble** | Soft-voting ensemble of XGBoost + LightGBM (Phase 2) |
| **Training date** | May 2025 |
| **Framework** | XGBoost 2.x, scikit-learn 1.x, imbalanced-learn (SMOTE) |
| **MLflow experiment** | `finshield_classical_ml` |
| **Registry name** | `finshield_fraud_detector` |
| **Primary contact** | FinShield project, portfolio submission |

### Hyperparameters (XGBoost tuned — Optuna, 100 trials)

| Parameter | Value |
|-----------|-------|
| n_estimators | 429 |
| max_depth | 5 |
| learning_rate | 0.0833 |
| subsample | 0.7639 |
| colsample_bytree | 0.9066 |
| scale_pos_weight | 577 |

---

## 2. Intended Use

### Primary Use Case
Real-time binary classification of financial transactions as **fraud** (Class=1) or
**legitimate** (Class=0). Designed to assist fraud analysts at banks and fintech companies
by surfacing high-risk transactions for human review.

### Intended Users
- Fraud operations teams at digital payment companies (e.g. Razorpay, PhonePe)
- Risk management teams at commercial banks (e.g. HDFC, ICICI)
- Data science teams building fraud detection pipelines

### Out-of-Scope Uses
The following uses are explicitly **not supported**:
- **Credit scoring** — this model predicts transaction fraud, not creditworthiness
- **Loan approval** — binary fraud labels do not correlate with repayment risk
- **Identity verification** — the dataset contains no PII or biometric features
- **Anti-money laundering (AML)** — requires graph-level analysis, not transaction-level
- **Regulatory enforcement action** — model output alone is insufficient for legal action

---

## 3. Training Data

| Attribute | Detail |
|-----------|--------|
| **Dataset** | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Source** | ULB Machine Learning Group (Université Libre de Bruxelles) |
| **Transactions** | 284,807 total |
| **Fraud cases** | 492 (0.172%) |
| **Time span** | 2 days of European cardholder transactions (September 2013) |
| **Geography** | European cardholders (country not specified) |

### Features

| Feature group | Count | Description |
|---------------|-------|-------------|
| PCA components | 28 (V1–V28) | Anonymised via PCA to protect cardholder privacy |
| Time | 1 | Seconds elapsed since first transaction |
| Amount | 1 | Transaction amount in EUR |
| Engineered features | 12 | Hour, Day, Is_night, Is_weekend, Amount_log, Amount_zscore, Is_round_amount, Is_small_amount, V14_V4_interaction, V14_V10_interaction, V14_squared, V10_squared |

### Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied to training set only
- `scale_pos_weight=577` in XGBoost to penalise missed fraud heavily
- Test set preserved at natural 0.172% fraud rate for honest evaluation

---

## 4. Evaluation Results

All metrics computed on a held-out test set (20% stratified split, 56,962 transactions,
~98 fraud cases).

### XGBoost Tuned (Primary Model)

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.8725** |
| **F1 Score** | 0.8300 |
| **Precision** | 0.8642 |
| **Recall** | 0.7959 |
| **ROC-AUC** | 0.9741 |
| **MCC** | 0.8291 |

### Ensemble (XGBoost + LightGBM soft-vote)

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.8756** |
| **F1 Score** | 0.8705 |

### Baseline Comparison

| Model | PR-AUC | F1 |
|-------|--------|----|
| Logistic Regression (baseline) | ~0.70 | ~0.65 |
| XGBoost tuned | 0.8725 | 0.8300 |
| **Ensemble (production)** | **0.8756** | **0.8705** |

### Operating Threshold Guidance

| Threshold | Precision | Recall | Use case |
|-----------|-----------|--------|----------|
| 0.30 | ~0.75 | ~0.92 | High-recall: catch more fraud, more false positives |
| 0.50 | ~0.86 | ~0.80 | Balanced operations |
| 0.70 | ~0.95 | ~0.65 | High-precision: fewer false positives, miss some fraud |

> **Note on deep learning**: A BiLSTM sequential model (Phase 3) achieved PR-AUC 0.6880.
> The lower score is expected — the dataset has no user IDs, so sequential fraud
> patterns cannot be properly modelled. Classical boosting remains state-of-the-art
> for flat anonymised transaction profiles.

---

## 5. Limitations

### Dataset Limitations
- **Geography**: Trained exclusively on European cardholder data from 2013. Performance
  on Indian UPI/mobile payments, NEFT/RTGS transactions, or post-2020 fraud patterns
  has **not been validated**.
- **Anonymisation**: V1–V28 are PCA-transformed. Domain experts cannot interpret
  which real features (merchant category, device fingerprint, etc.) drive predictions.
- **No user identity**: All transactions are independent — the model cannot detect
  account takeover sequences or velocity-based fraud across multiple transactions.
- **Time scope**: 2-day snapshot. Seasonal fraud patterns (e.g. Diwali, Christmas
  shopping spikes) are not represented in training data.
- **Currency**: Amounts are in EUR. Indian INR transaction distributions are significantly
  different and may require recalibration.

### Model Limitations
- The model produces a **probability score**, not a deterministic fraud verdict.
  Score calibration should be validated before setting operational thresholds.
- Adversarial fraud (where fraudsters deliberately mimic legitimate patterns) may
  not be detected until the model is retrained on new labelled examples.

---

## 6. Ethical Considerations

### False Positive Cost
Every false positive (a legitimate transaction flagged as fraud) results in:
- Customer transaction decline → friction and potential churn
- Customer support escalation cost (~₹500–₹2,000 per incident)
- Potential discrimination if false positive rates differ across demographic groups

**Recommendation**: Monitor false positive rates segmented by transaction amount bucket,
merchant category, and time-of-day. Alert if any segment exceeds 2× the global false
positive rate.

### Human Review Requirement
- Model output should **not** be used as the sole basis for blocking a customer account
- Transactions with scores in the **0.40–0.60 range** (borderline zone) should be
  routed to a human fraud analyst for review before action is taken
- High-score transactions (>0.75) may be auto-declined, but the customer must be
  notified and provided a clear escalation path

### Bias and Fairness
- The training dataset contains no demographic features (age, gender, nationality)
- However, spending patterns encoded in V1–V28 may implicitly correlate with
  demographic attributes — this risk cannot be fully quantified without the
  original un-anonymised data
- **Action required**: Before production deployment, conduct disparate impact analysis
  using proxy variables (transaction geolocation, merchant category distribution)

### Regulatory Compliance
- RBI Master Circular on Fraud Risk Management (2023) mandates human oversight for
  automated fraud decisions above ₹1 lakh
- PCI DSS v4.0 Requirement 10 mandates audit trails for all fraud decision events
- All model predictions should be logged with timestamp, feature hash, and score
  for regulatory audit readiness

---

## 7. Drift and Maintenance

### Automated Monitoring (Phase 5)
| Component | Tool | Frequency |
|-----------|------|-----------|
| Feature drift detection | Evidently AI | Weekly batch |
| Model performance monitoring | Custom (sklearn metrics) | Daily batch simulation |
| Experiment tracking | MLflow | Per training run |
| Pipeline orchestration | Prefect | On-demand + scheduled |

### Retraining Triggers
- **Automated**: Evidently drift share > 30% of features → triggers `retrain_model()`
- **Scheduled**: Monthly full retraining regardless of drift
- **Manual**: Data science team discretion after incident review

### Recommended Review Cadence
- **Weekly**: Review drift report HTML (`data/drift_reports/`)
- **Monthly**: Full model evaluation on new labelled data
- **Quarterly**: Stakeholder review of threshold settings and false positive rates
- **Annually**: Full dataset refresh and architecture review

---

## 8. Citation

If using this model or codebase in research or production:

```
@misc{finshield2025,
  title  = {FinShield: Production-Grade AI Fraud Detection Platform},
  year   = {2025},
  note   = {Portfolio project. XGBoost ensemble trained on Kaggle Credit Card Fraud dataset.
            MLflow + Evidently AI + Prefect MLOps stack.}
}
```

Dataset citation:
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
```
