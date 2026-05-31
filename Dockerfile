FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY static/ static/
COPY scripts/ scripts/
COPY entrypoint.sh .

ENV ASR_MODEL_DIR=/app/models \
    ASR_MODEL=en-meta \
    ASR_DEVICE=cpu \
    PORT=8001 \
    HOST=0.0.0.0 \
    HF_TOKEN=""

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
