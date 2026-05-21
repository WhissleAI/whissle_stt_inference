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

ENV ASR_MODEL_DIR=/app/models
ENV PORT=8001
ENV HOST=0.0.0.0

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001"]
