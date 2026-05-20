"""Dual-backend embedder for runtime query embedding.

Mirrors the actor's design so the same model can be used compile-time and
query-time (cosine similarity only makes sense in the same vector space).

Model name conventions:
- ``pinecone:<model>``   → cloud via Pinecone Inference (requires PINECONE_API_KEY)
- ``<huggingface_repo>`` → local CPU via sentence-transformers
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np

from ._logging import get_logger

log = get_logger("embedder")

_PINECONE_BATCH_LIMIT = 96


class EmbedderConfigError(RuntimeError):
    """Raised when an embedder is misconfigured (missing key, missing extra, etc.)."""


class Embedder:
    """Lazy dual-backend embedder. Pick backend from model name prefix.

    The model is loaded on first use so the MCP server boots fast even when the
    user only ever calls metadata tools (e.g. ``list_wiki_pages``).
    """

    def __init__(self, model_name: str = "pinecone:multilingual-e5-large"):
        self.model_name = model_name
        if model_name.startswith("pinecone:"):
            self.backend: Literal["pinecone", "local"] = "pinecone"
            self.remote_model: str | None = model_name.split(":", 1)[1]
        else:
            self.backend = "local"
            self.remote_model = None
        self._st_model = None
        self._pc_inference = None

    # ── local ──────────────────────────────────────────────────────────────
    def _ensure_local(self) -> None:
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderConfigError(
                f"local backend requested (model={self.model_name!r}) but "
                "sentence-transformers is not installed. "
                "Reinstall with: pip install 'wiki-embedded-mcp[local]'"
            ) from e
        log.info("loading local embedder: %s", self.model_name)
        self._st_model = SentenceTransformer(self.model_name)

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        self._ensure_local()
        return self._st_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    # ── pinecone ───────────────────────────────────────────────────────────
    def _ensure_pinecone(self) -> None:
        if self._pc_inference is not None:
            return
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise EmbedderConfigError(
                f"PINECONE_API_KEY env var required for embedder={self.model_name!r}. "
                "Set it on the actor / Claude Desktop config, or switch to a local model "
                "(e.g. 'intfloat/multilingual-e5-base') with the [local] extra."
            )
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise EmbedderConfigError(
                f"pinecone backend requested (model={self.model_name!r}) but the "
                "pinecone client is not installed. "
                "Reinstall with: pip install 'wiki-embedded-mcp[cloud]'"
            ) from e
        log.info("initializing Pinecone inference client (model=%s)", self.remote_model)
        self._pc_inference = Pinecone(api_key=api_key).inference

    def _embed_pinecone(self, texts: list[str], input_type: str) -> np.ndarray:
        self._ensure_pinecone()
        # Pinecone applies its own E5 prefix when given input_type, so strip ours.
        clean = [t.split(": ", 1)[1] if t.startswith(("passage:", "query:")) else t for t in texts]
        vectors: list[list[float]] = []
        for i in range(0, len(clean), _PINECONE_BATCH_LIMIT):
            batch = clean[i : i + _PINECONE_BATCH_LIMIT]
            try:
                resp = self._pc_inference.embed(
                    model=self.remote_model,
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
            except Exception as e:
                # Wrap any Pinecone client exception with a clearer message
                raise EmbedderConfigError(
                    f"Pinecone inference call failed (model={self.remote_model!r}): {e}"
                ) from e
            for item in resp:
                vec = item.values if hasattr(item, "values") else item["values"]
                vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)

    # ── public ─────────────────────────────────────────────────────────────
    def embed_query(self, text: str) -> np.ndarray:
        if self.backend == "pinecone":
            return self._embed_pinecone([text], input_type="query")[0]
        return self._embed_local([f"query: {text}"])[0]

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if self.backend == "pinecone":
            return self._embed_pinecone(texts, input_type="passage")
        return self._embed_local([f"passage: {t}" for t in texts])
