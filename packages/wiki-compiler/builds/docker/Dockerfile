# wiki-compiler — Standalone Docker image (no Apify SDK)
#
# Use this for self-hosted compile jobs or CI-based wiki builds.
# Build context: packages/wiki-compiler/
#   docker build -f builds/docker/Dockerfile -t wiki-compiler:local .
#
# Run:
#   docker run --rm \
#     -e GOOGLE_API_KEY=... \
#     -v $PWD/my-wiki:/wiki:ro \
#     -v $PWD/out:/out \
#     wiki-compiler:local \
#     --source /wiki --out /out --intent "What this wiki is for"

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

# CPU-only torch
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

# Project
COPY --chown=app:app pyproject.toml requirements.txt README.md ./
RUN pip install --user -r requirements.txt

# Pre-cache E5 model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Source + editable install
COPY --chown=app:app src/ ./src/
RUN pip install --user -e .

ENTRYPOINT ["wiki-compile"]
