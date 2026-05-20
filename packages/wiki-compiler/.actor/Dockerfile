# wiki-compiler — Apify Actor Dockerfile
#
# Build context: packages/wiki-compiler/
# Apify CLI:    `apify push` from that directory
#
# Cold-start optimizations:
# - Install CPU-only torch (saves ~5GB CUDA toolkit, 70-80% faster cold start)
# - Pre-cache the default E5 multilingual model into the image
# - Layer requirements.txt before source COPY so pip cache is hit on src-only changes

FROM apify/actor-python:3.11

# 1. CPU-only torch first (so sentence-transformers reuses it)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Project metadata + requirements (layer-friendly)
COPY --chown=myuser:myuser pyproject.toml requirements.txt README.md ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-download default E5 model (saves ~500MB per cold start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')" \
    && python -c "import torch; print('torch OK', torch.__version__, 'cuda?', torch.cuda.is_available())"

# 4. Project source + editable install
COPY --chown=myuser:myuser src/ ./src/
COPY --chown=myuser:myuser .actor/ ./.actor/
RUN pip install --no-cache-dir -e .

# Run the Actor
CMD ["python", "-m", "wiki_compiler"]
