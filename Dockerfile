# syntax=docker/dockerfile:1.7
#
# wiki-embedded-mcp — MCP server runtime image.
#
# Two variants are built from the same Dockerfile:
#   `:latest`         (default) — cloud-only backend (Pinecone). ~150 MB. Fast cold start.
#   `:latest-full`    — bundles sentence-transformers + torch for offline use. ~1.2 GB.
#
# Build the lightweight default:
#   docker build -t wiki-embedded-mcp:latest .
#
# Build the full local-embedder image:
#   docker build --build-arg EXTRA=all -t wiki-embedded-mcp:latest-full .
#
# Run (Claude Desktop / Cursor / Cline talk to it over stdio):
#   docker run -i --rm -e PINECONE_API_KEY ghcr.io/42rows/wiki-embedded-mcp:latest \
#       --compiled-wiki https://.../compiled_wiki.zip

ARG EXTRA=cloud

# ── builder ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
ARG EXTRA
WORKDIR /build

# System deps only for the build (compilers gone in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install into a target dir so the runtime stage can COPY it cleanly
RUN pip install --no-cache-dir --target=/install ".[${EXTRA}]"

# Optional: pre-cache the default local E5 model into the image when building
# the `full` variant so cold starts don't pay the 500 MB HuggingFace download.
RUN if [ "$EXTRA" = "all" ] || [ "$EXTRA" = "local" ]; then \
        PYTHONPATH=/install python -c \
        "from sentence_transformers import SentenceTransformer; \
         SentenceTransformer('intfloat/multilingual-e5-base')"; \
    fi

# ── runtime ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user (FAANG security baseline)
RUN useradd -m -u 1000 -s /usr/sbin/nologin mcp
WORKDIR /app

# Copy installed packages + any cached HF models
COPY --from=builder /install /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.cache /home/mcp/.cache
RUN chown -R mcp:mcp /home/mcp/.cache || true

USER mcp

# MCP servers must keep stdout reserved for JSON-RPC; logs go to stderr by default.
ENV PYTHONUNBUFFERED=1 \
    WIKI_EMBEDDED_LOG_LEVEL=INFO

ENTRYPOINT ["python", "-m", "wiki_embedded_mcp.server"]
