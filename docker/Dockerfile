FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /cadence

# System deps for LightGBM (libgomp) and MLflow's git integration.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY cadence ./cadence
COPY configs ./configs

RUN pip install --upgrade pip && pip install -e .[ml,drift,api,dev]

# Torch-Geometric (Phase 3+) is left for a separate GPU Dockerfile because its
# torch-scatter/torch-sparse wheels are CUDA/OS-specific.

EXPOSE 8000
CMD ["cadence", "--help"]
