# polars-runner — Apify Actor Dockerfile
#
# Build context: packages/polars-runner/
# Apify CLI:    `apify push` from that directory
#
# Build-cache optimization:
# - pyproject.toml + requirements.txt copied BEFORE src/ for layer reuse
# - non-root `myuser` per Apify 2026 best practice (--chown on every COPY)

FROM apify/actor-python:3.11

# 1. Project metadata + requirements (layer-friendly)
COPY --chown=myuser:myuser pyproject.toml requirements.txt README.md ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Source + editable install
COPY --chown=myuser:myuser src/ ./src/
COPY --chown=myuser:myuser .actor/ ./.actor/
RUN pip install --no-cache-dir -e .

# Run the Actor
CMD ["python", "-m", "polars_runner"]
