# FinShield - AI-Powered Financial Fraud Detection Platform

[![CI](https://github.com/Ryan-prog2006/Financial-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryan-prog2006/Financial-Fraud-Detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> End-to-end fraud detection platform combining XGBoost, BiLSTM deep learning, LLM+RAG intelligence, and production MLOps — deployed as a live demo.

**[🚀 Live Demo](https://financial-fraud-detection-a1.streamlit.app/)** · **[📖 Local API Docs](http://localhost:8000/docs)** · **[📊 Local MLflow UI](http://localhost:5000)**

---

## The Problem

In modern digital banking and card-not-present environments, financial fraud occurs at an unprecedented scale and speed. Standard rule-based fraud detection engines are static, reactive, and easily bypassed by sophisticated fraudsters, costing financial institutions billions of dollars globally and introducing immense operational friction.

Traditional machine learning classifiers often fail because transaction fraud is highly imbalanced (under 0.2% of transactions are fraudulent) and exhibits temporal patterns that require sequential reasoning. Furthermore, fraud analysts cannot rely on black-box predictions; they must explain why a transaction was flagged and quickly check if it violates specific compliance frameworks like RBI directives or PCI DSS rules.

FinShield solves this problem using a 5-layer intelligence architecture. It combines classical machine learning ensemble models optimized for highly imbalanced distributions, a PyTorch BiLSTM neural network for sequential transaction screening, a LangChain RAG pipeline grounded in compliance docs, and production-grade MLOps pipelines with data drift detection.

## Architecture

Transaction Input
│
▼
┌─────────────────────────────────────────────────┐
│  Layer 1: Feature Engineering Pipeline          │
│  (15+ features: velocity, z-score, time, etc.)  │
└──────────────────────┬──────────────────────────┘
│
┌─────────────────┴──────────────────┐
▼                                    ▼
┌──────────────┐                  ┌──────────────┐
│  Layer 2:    │                  │  Layer 3:    │
│  XGBoost +   │                  │  BiLSTM      │
│  LightGBM    │                  │  (sequences) │
│  Ensemble    │                  └──────┬───────┘
└──────┬───────┘                         │
└───────────────┬─────────────────┘
▼
┌──────────────────────┐
│  Layer 4: LLM + RAG  │
│  (LangChain + Groq)  │
│  Fraud explanation + │
│  Policy citation     │
└──────────┬───────────┘
│
┌──────────▼───────────┐
│  Layer 5: MLOps      │
│  MLflow + Evidently  │
│  Auto-retraining     │
└──────────────────────┘

## Results

Below are the metric results obtained during model training and evaluation:

| Model | PR-AUC | F1 Score | Precision | Recall | Type |
|---|---|---|---|---|---|
| Ensemble (Production) | 0.8756 | 0.8705 | 0.8966 | 0.8455 | Ensemble |
| XGBoost (Tuned) | 0.8725 | 0.8300 | 0.8137 | 0.8469 | Classical |
| LightGBM (Tuned) | 0.8723 | 0.8723 | 0.8723 | 0.8723 | Classical |
| Random Forest | 0.8312 | 0.8201 | 0.8115 | 0.8288 | Classical |
| Logistic Regression | 0.7021 | 0.6512 | 0.6122 | 0.6954 | Baseline |
| BiLSTM (Deep Learning) | 0.6880 | 0.1762 | 0.1012 | 0.6890 | Deep Learning |

## Project Structure

```
finshield/
├── .github/
│   └── workflows/     # GitHub Actions CI/CD workflows
├── dashboard/         # Streamlit dashboard files
├── data/
│   ├── raw/           # Original CSV transaction data (never modified)
│   └── processed/     # Processed and engineered datasets
├── docker/            # Dockerfiles for API and MLflow servers
├── hf_app/            # Self-contained Hugging Face Spaces app
├── notebooks/
│   └── figures/       # EDA and model monitoring charts
├── src/
│   ├── api/           # FastAPI REST API source code
│   ├── data/          # Feature engineering and validation pipeline
│   ├── llm/           # RAG and Groq LLM integration
│   ├── models/        # Classical and deep learning pipelines
│   └── mlops/         # Drift detection and retrain workflows
├── tests/             # Pytest unit tests for all phases
├── requirements.txt   # Main dependency requirements file
├── docker-compose.yml # Docker Compose services configuration
├── Makefile           # Task automation shortcuts
├── AGENTS.md          # Project identity and stack definition
└── README.md          # Project documentation
```

## Quick Start (local)

Set up and run the FinShield project locally by following these steps:

```bash
# Clone the repository
git clone https://github.com/Ryan-prog2006/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection

# Create virtual environment and activate
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env  # Add your GROQ_API_KEY to this file

# Run the complete data engineering pipeline
python src/data/pipeline.py

# Train models and register them
python src/models/pipeline_ml.py

# Build RAG system and vector database
python src/llm/pipeline_llm.py

# Run the FastAPI server locally
uvicorn src.api.main:app --port 8000

# Start the Streamlit analyst dashboard (in a separate terminal)
streamlit run dashboard/app.py --server.port 8501

# Alternatively, run the full services stack using Docker Compose
docker-compose up --build
```

## Tech Stack

- **Machine Learning**: XGBoost, LightGBM, Scikit-learn, Imbalanced-learn (SMOTE), Optuna.
- **Deep Learning**: PyTorch (BiLSTM, CNN1D).
- **LLM & RAG**: LangChain, ChromaDB, Hugging Face Embeddings, Groq API (Llama 3).
- **MLOps & Pipelines**: MLflow, Evidently AI, Prefect, Docker, Docker Compose.
- **Serving & UI**: FastAPI, Pydantic v2, Streamlit, Plotly.
- **Testing & CI/CD**: Pytest, GitHub Actions.

## Phase Breakdown

- **Phase 1**: Data Engineering & EDA. Pandera schemas, custom feature extraction, and class imbalance mitigation.
- **Phase 2**: Classical ML. Tuned XGBoost + LightGBM classifiers with MCC/PR-AUC optimization.
- **Phase 3**: Deep Learning. PyTorch sequence builder and BiLSTM classification model.
- **Phase 4**: LLM + RAG. ChromaDB vector database creation, LangChain retriever, and RAGAS evaluations.
- **Phase 5**: MLOps. Automated drift reports, model registry promotion, and Docker containerization.
- **Phase 6**: API & Dashboard. Rate-limited FastAPI application and Streamlit monitoring front-end.
- **Phase 7**: Deployment & CI/CD. GitHub Actions pipelines and standalone Hugging Face Space app.

## API Reference

### Health check
```bash
curl http://localhost:8000/health
```

### Predict fraud
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 847.23, "time_hour": 2.3, "is_night": 1, "is_weekend": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34, "V6": -0.47, "V7": 0.21, "V8": 0.10, "V9": -0.62, "V10": -0.99, "V11": -0.31, "V12": -1.16, "V13": -0.47, "V14": -0.64, "V15": -0.22, "V16": -1.16, "V17": -2.13, "V18": -0.10, "V19": 0.14, "V20": -0.45, "V21": -0.23, "V22": -0.53, "V23": 0.25, "V24": 0.04, "V25": 0.13, "V26": -0.04, "V27": 0.04, "V28": -0.01}'
```

### Get model metadata
```bash
curl http://localhost:8000/model/info
```

## Targeting Indian Fintech Roles

FinShield is specifically aligned with the security and compliance requirements of the Indian fintech and banking sector. The RAG document repository indexes the Reserve Bank of India (RBI) Master Directions on Digital Payment Security Controls and Fraud Risk Management. LLM outputs are trained to refer to Indian banking mandates, such as reporting thresholds and UPI transaction velocity limits, matching the domain requirements of fintech companies like Razorpay, PhonePe, and HDFC Bank.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
