"""Embedder with dual backend: Pinecone Inference (cloud) or sentence-transformers (local).

Model name conventions:
- ``pinecone:<model>``   → cloud via Pinecone Inference (requires PINECONE_API_KEY)
- ``<huggingface_repo>`` → local CPU via sentence-transformers

E5 family requires ``passage:`` / ``query:`` prefixes; we apply ``passage:`` for
indexed content and the consumer applies ``query:`` at retrieval time.

Rate-limit handling (Pinecone Inference free tier = 250K tokens / minute):
- Proactive token-budget throttling between batches.
- Exponential backoff retry on HTTP 429 / RESOURCE_EXHAUSTED.
- Bounded backoff (max 3 retries per batch).
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EmbeddingResult:
    model_name: str
    dims: int
    vectors: np.ndarray  # (N, dims), float32
    slug_order: list[str]  # parallel to rows

    def to_dict_for_manifest(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dims": self.dims,
            "vectors_count": int(self.vectors.shape[0]),
            "slug_order": self.slug_order,
        }


# Pinecone Inference limits (free tier defaults — May 2026).
_PINECONE_BATCH_LIMIT = 96                  # inputs per request
_PINECONE_TOKENS_PER_MIN = 240_000          # 250K nominal, keep 4% buffer
_PINECONE_MAX_RETRIES = 4
_PINECONE_INITIAL_BACKOFF_S = 30.0          # 1st retry waits ~half a minute window
_PINECONE_MAX_BACKOFF_S = 120.0


def _estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate (~4 chars per token for English/Italian)."""
    return max(1, len(text) // 4)


def _is_rate_limit(err: BaseException) -> bool:
    """Best-effort detection of a Pinecone 429 / RESOURCE_EXHAUSTED."""
    msg = str(err).lower()
    if "429" in msg or "rate" in msg or "resource_exhausted" in msg or "max tokens per minute" in msg:
        return True
    # The pinecone SDK raises ApiException with .status / .reason on some paths
    status = getattr(err, "status", None) or getattr(err, "status_code", None)
    return status == 429


class E5Embedder:
    """Dual-backend embedder with rate-limit-aware batching.

    Resolves backend at construction time from `model_name`:
      - "pinecone:<model>" → Pinecone Inference HTTP API
      - anything else      → sentence-transformers local CPU
    """

    def __init__(self, model_name: str = "pinecone:multilingual-e5-large"):
        self.model_name = model_name
        if model_name.startswith("pinecone:"):
            self.backend = "pinecone"
            self.remote_model = model_name.split(":", 1)[1]
        else:
            self.backend = "local"
            self.remote_model = None
        self._st_model = None         # sentence-transformers lazy
        self._pc_inference = None     # pinecone inference lazy

        # Rolling window of (timestamp, tokens) for proactive throttling.
        self._token_window: list[tuple[float, int]] = []

    # ------------------------------------------------------------------ local
    def _ensure_local(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)

    def _embed_local(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        self._ensure_local()
        return self._st_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def _local_dims(self) -> int:
        self._ensure_local()
        return int(self._st_model.get_sentence_embedding_dimension())

    # --------------------------------------------------------------- pinecone
    def _ensure_pinecone(self) -> None:
        if self._pc_inference is None:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    f"PINECONE_API_KEY env var required for embeddingModel={self.model_name!r}. "
                    "Either set the env var on the actor or pick a local model."
                )
            from pinecone import Pinecone
            pc = Pinecone(api_key=api_key)
            self._pc_inference = pc.inference

    def _wait_under_budget(self, next_batch_tokens: int) -> None:
        """Sleep until the rolling 60-second token window can absorb the next batch."""
        now = time.time()
        # Drop entries older than 60s
        self._token_window = [(t, n) for (t, n) in self._token_window if now - t < 60.0]
        used = sum(n for _, n in self._token_window)
        budget = _PINECONE_TOKENS_PER_MIN
        if used + next_batch_tokens <= budget:
            return
        # We'd exceed budget — wait until the oldest entry rolls off.
        oldest_ts = self._token_window[0][0] if self._token_window else now
        sleep_s = max(1.0, 60.0 - (now - oldest_ts) + 1.0)
        print(
            f"[embed] proactive throttle: used={used} + next={next_batch_tokens} "
            f"> budget={budget}/min — sleeping {sleep_s:.0f}s"
        )
        time.sleep(sleep_s)
        # Recompute window after sleep
        now = time.time()
        self._token_window = [(t, n) for (t, n) in self._token_window if now - t < 60.0]

    def _embed_pinecone_batch(
        self, batch_clean: list[str], input_type: str
    ) -> list[list[float]]:
        """Single Pinecone Inference call with retry on 429."""
        delay = _PINECONE_INITIAL_BACKOFF_S
        last_err: BaseException | None = None
        for attempt in range(1, _PINECONE_MAX_RETRIES + 1):
            try:
                resp = self._pc_inference.embed(
                    model=self.remote_model,
                    inputs=batch_clean,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                vecs: list[list[float]] = []
                for item in resp:
                    v = item.values if hasattr(item, "values") else item["values"]
                    vecs.append(v)
                return vecs
            except Exception as e:  # noqa: BLE001 — Pinecone SDK raises bare exceptions
                last_err = e
                if not _is_rate_limit(e):
                    raise
                if attempt >= _PINECONE_MAX_RETRIES:
                    break
                wait_s = min(delay, _PINECONE_MAX_BACKOFF_S)
                print(
                    f"[embed] rate-limited (attempt {attempt}/{_PINECONE_MAX_RETRIES}), "
                    f"backing off {wait_s:.0f}s — {e!s}"
                )
                time.sleep(wait_s)
                delay *= 2  # exponential
        raise RuntimeError(
            f"Pinecone Inference rate-limited after {_PINECONE_MAX_RETRIES} retries: {last_err}"
        )

    def _embed_pinecone(self, texts: list[str], input_type: str = "passage") -> np.ndarray:
        self._ensure_pinecone()
        # Strip our local E5 prefix — Pinecone applies its own when input_type is set.
        clean = [t.split(": ", 1)[1] if t.startswith(("passage:", "query:")) else t for t in texts]

        vectors: list[list[float]] = []
        for i in range(0, len(clean), _PINECONE_BATCH_LIMIT):
            batch = clean[i : i + _PINECONE_BATCH_LIMIT]
            batch_tokens = sum(_estimate_tokens(t) for t in batch)
            self._wait_under_budget(batch_tokens)

            batch_vecs = self._embed_pinecone_batch(batch, input_type)
            vectors.extend(batch_vecs)

            # Record the call for the rolling token-budget window.
            self._token_window.append((time.time(), batch_tokens))

        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------------ public
    def embed_pages(
        self,
        pages: list[dict[str, Any]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> EmbeddingResult:
        slugs: list[str] = []
        texts: list[str] = []
        for p in pages:
            text = (p.get("content_full") or "").strip()
            if not text:
                continue
            slugs.append(p["slug"])
            texts.append(f"passage: {text}")

        if not texts:
            dims = self._local_dims() if self.backend == "local" else 1024
            return EmbeddingResult(
                model_name=self.model_name,
                dims=dims,
                vectors=np.zeros((0, dims), dtype=np.float32),
                slug_order=[],
            )

        t0 = time.time()
        if self.backend == "pinecone":
            vectors = self._embed_pinecone(texts, input_type="passage")
        else:
            vectors = self._embed_local(texts, batch_size, show_progress)
        elapsed = time.time() - t0
        print(
            f"[embed] backend={self.backend} model={self.model_name} "
            f"count={len(texts)} dims={vectors.shape[1]} elapsed={elapsed:.1f}s"
        )

        return EmbeddingResult(
            model_name=self.model_name,
            dims=int(vectors.shape[1]),
            vectors=vectors,
            slug_order=slugs,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Single-text embed as a query vector (for at-compile-time retrieval)."""
        if self.backend == "pinecone":
            vec = self._embed_pinecone([text], input_type="query")[0]
        else:
            vec = self._embed_local([f"query: {text}"], batch_size=1, show_progress=False)[0]
        return vec
