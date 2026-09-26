# FraudGuard Streamlit dashboard image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUDGUARD_ROOT=/app \
    FRAUDGUARD_MODELS_DIR=/app/models \
    FRAUDGUARD_DATABASE_URL=sqlite:////app/data/db/fraudguard.db \
    FRAUDGUARD_API_URL=http://api:8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install -r requirements.txt && pip install -e .

COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY models ./models
COPY reports ./reports
RUN mkdir -p /app/data/db

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
