FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[serve]"

COPY src/ src/

# Model bundle is mounted at runtime via volume (see docker-compose.yml)
ENV MODEL_BUNDLE_PATH=/app/model/best_model_bundle.pkl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
