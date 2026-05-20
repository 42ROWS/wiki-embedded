# polars-runner — Standalone Docker image (no Apify SDK)
#
# Use this for self-hosted transformations or CI-based data pipelines.
# Build context: packages/polars-runner/
#   docker build -f builds/docker/Dockerfile -t polars-runner:local .
#
# Run:
#   docker run --rm \
#     -e GOOGLE_API_KEY=... \
#     -v $PWD/data:/data:ro \
#     polars-runner:local \
#     --input /data/input.json --prompt "Group by sector, sum revenue"

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

ENV PATH="/home/app/.local/bin:${PATH}"

# Project metadata + requirements (layer-friendly)
COPY --chown=app:app pyproject.toml requirements.txt README.md ./
RUN pip install --user -r requirements.txt

# Source + editable install (with all optional extras)
COPY --chown=app:app src/ ./src/
RUN pip install --user -e ".[all]"

ENTRYPOINT ["polars-runner"]
