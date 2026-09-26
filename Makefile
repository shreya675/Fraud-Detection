.PHONY: install install-dev data preprocess train evaluate api dashboard test lint docker-build docker-up docker-down clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

data:
	$(PYTHON) scripts/download_data.py

preprocess:
	$(PYTHON) -m fraudguard.data

train:
	$(PYTHON) -m fraudguard.train

evaluate:
	$(PYTHON) -m fraudguard.evaluate

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} +
