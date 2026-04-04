# ============================================================
# Stage 1 — builder: install dependencies into a venv
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt \
 && pip install --prefix=/install --no-cache-dir gradio>=4.0.0


# ============================================================
# Stage 2 — runtime: lean production image
# ============================================================
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/Ikteaja/sms-spam-transformer"
LABEL org.opencontainers.image.description="SMS Spam Classifier — DistilBERT transfer learning"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/       ./app/
COPY scripts/   ./scripts/

# Model is mounted at runtime via Docker volume or downloaded on first start.
# To bake in a fine-tuned model:
#   docker build --build-arg MODEL_SRC=models/frozen .
ARG MODEL_SRC=""
COPY ${MODEL_SRC:-app} ./models_baked/
# (If MODEL_SRC is not set the COPY above is a no-op for model files)

# Non-root user for security
RUN useradd -m -u 1000 appuser \
 && mkdir -p models data \
 && chown -R appuser:appuser /app
USER appuser

# Environment
ENV MODEL_DIR=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false

# FastAPI on 8000, Gradio mounted at /ui (same port)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
