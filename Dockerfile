# syntax=docker/dockerfile:1

# ---------- Étape 1 : installation des dépendances ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------- Étape 2 : image finale, légère et sans outils de build ----------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="prediction-defaut-credit" \
      org.opencontainers.image.description="API de scoring du risque de défaut de paiement" \
      org.opencontainers.image.source="https://github.com/Naomi-na2024/prediction-defaut-credit-mlops" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    WEB_CONCURRENCY=2 \
    LOG_LEVEL=INFO

# L'application ne tourne pas en root (bonne pratique de sécurité)
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY credit_risk/ credit_risk/
COPY models/ models/

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)" || exit 1

CMD ["sh", "-c", "exec uvicorn credit_risk.api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --no-server-header --no-access-log"]
