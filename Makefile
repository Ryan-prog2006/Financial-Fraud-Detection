.PHONY: install test run-api run-dashboard run-docker setup-hf clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run-api:
	uvicorn src.api.main:app --reload --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

run-docker:
	docker-compose up --build

pipeline-all:
	python src/data/pipeline.py
	python src/models/pipeline_ml.py
	python src/llm/pipeline_llm.py
	python src/mlops/pipeline_mlops.py

setup-hf:
	python hf_app/setup_artifacts.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
