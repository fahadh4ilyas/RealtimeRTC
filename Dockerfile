# syntax=docker/dockerfile:1

# ── CPU-only override:  docker build --build-arg BASE=python:3.11-slim .
# ── Default (CUDA 12):  docker build .
ARG BASE=nvidia/cuda:12.4.0-runtime-ubuntu22.04

FROM ${BASE}

# ------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Python venv — Ubuntu disables system ensurepip/pip, so create a
# self-contained venv (with pip) and run everything from it.
# ------------------------------------------------------------------
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ------------------------------------------------------------------
# Python dependencies (cached layer)
# ------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# Application code
# ------------------------------------------------------------------
COPY realtimertc/ realtimertc/

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE=auto

EXPOSE 8081

CMD ["python", "-m", "realtimertc.main", "--host", "0.0.0.0", "--port", "8081"]
