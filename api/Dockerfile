# FraudGuard API image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUDGUARD_ROOT=/app \
    FRAUDGUARD_MODELS_DIR=/app/models \
    FRAUDGUARD_DATABASE_URL=sqlite:////app/data/db/fraudguard.db

WORKDIR /app

# libgomp is required by xgboost's OpenMP build
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install -r requirements.txt && pip install -e .

COPY api ./api
COPY models ./models
COPY reports ./reports
RUN mkdir -p /app/data/db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
