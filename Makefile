.PHONY: install train train-mlflow test lint format run docker docker-run

install:
	pip install -r requirements-dev.txt

train:
	python -m credit_risk.train

train-mlflow:
	python -m credit_risk.train --mlflow

test:
	pytest --cov --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

run:
	uvicorn credit_risk.api:app --reload --port 5000

docker:
	docker compose up --build

docker-run:
	docker build -t prediction-defaut-credit . && docker run --rm -p 5000:5000 prediction-defaut-credit
