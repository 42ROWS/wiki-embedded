"""Embedder-pluggable retrieval index with optional precomputed embeddings.

Two modes:
- ``fresh``        — embed pages on first query using sentence-transformers (slow first call)
- ``precomputed`` — use vectors loaded from a compiled_wiki.zip (instant)

Cosine search only — for production at scale (>50K pages), wire FAISS IVF on top.
"""
from __future__ import annotations

import os

import numpy as np

from ._logging import get_logger
from .embedder import Embedder

log = get_logger("index")

DEFAULT_EMBEDDER = "intfloat/multilingual-e5-base"
# Max chars per passage encoded at compile-fresh time. Override via env if your
# pages routinely exceed this and you accept the latency hit.
MAX_CHARS = int(os.getenv("WIKI_EMBEDDED_MAX_PASSAGE_CHARS", "1500"))


class WikiIndex:
    """In-memory wiki index: embeddings + cosine retrieval."""

    def __init__(
        self,
        compiled: dict,
        embedder_name: str = DEFAULT_EMBEDDER,
    ):
        self.compiled = compiled
        self.pages = compiled["pages"]
        self.slug_set = compiled["slug_set"]
        self.embedder_name = embedder_name
        self.page_emb: np.ndarray | None = None
        self._embedder: Embedder | None = None
        self.slug_to_idx = {p["slug"]: i for i, p in enumerate(self.pages)}
        # Parallel slug order to page_emb rows (set by either embed_pages or set_precomputed)
        self.emb_slug_order: list[str] | None = None

    # ── embedder ───────────────────────────────────────────────────────────
    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.embedder_name)
        return self._embedder

    # ── fresh embed ────────────────────────────────────────────────────────
    def embed_pages(self) -> None:
        """Encode all pages (idempotent). Used when no precomputed embeddings."""
        if self.page_emb is not None:
            return
        log.info("fresh-embedding %d pages with model=%s", len(self.pages), self.embedder_name)
        emb = self._get_embedder()
        slugs = [p["slug"] for p in self.pages]
        texts = [(p.get("content_full") or "")[:MAX_CHARS] for p in self.pages]
        self.page_emb = emb.embed_passages(texts)
        self.emb_slug_order = slugs
        log.info("embedded %d pages (dims=%d)", self.page_emb.shape[0], self.page_emb.shape[1])

    # ── precomputed ────────────────────────────────────────────────────────
    def set_precomputed(self, vectors: np.ndarray, slug_order: list[str]) -> None:
        """Inject precomputed embeddings (from a compiled_wiki.zip).

        Vectors are assumed L2-normalized (cosine reduces to dot product).
        """
        if vectors.shape[0] != len(slug_order):
            raise ValueError(
                f"vectors rows ({vectors.shape[0]}) != slug_order length ({len(slug_order)})"
            )
        self.page_emb = vectors.astype(np.float32)
        self.emb_slug_order = list(slug_order)
        log.info("loaded %d precomputed embeddings (dims=%d)", vectors.shape[0], vectors.shape[1])

    # ── query ──────────────────────────────────────────────────────────────
    def embed_query(self, query: str) -> np.ndarray:
        return self._get_embedder().embed_query(query)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Cosine search → top-K (slug, score) sorted desc."""
        if self.page_emb is None:
            self.embed_pages()
        q = self.embed_query(query)
        sims = self.page_emb @ q
        order = self.emb_slug_order or [p["slug"] for p in self.pages]
        top_idx = np.argsort(-sims)[:top_k]
        return [(order[i], float(sims[i])) for i in top_idx if i < len(order)]

    def get_page(self, slug: str) -> dict | None:
        idx = self.slug_to_idx.get(slug)
        return self.pages[idx] if idx is not None else None

    # ── graph queries ──────────────────────────────────────────────────────
    def get_backlinks(self, slug: str) -> list[str]:
        """Return slugs that cite the given slug (reverse crossref graph)."""
        graph = self.compiled.get("graph") or {}
        return sorted(s for s, refs in graph.items() if slug in refs)

    def get_forward_links(self, slug: str) -> list[str]:
        """Return slugs cited by the given slug."""
        graph = self.compiled.get("graph") or {}
        return sorted(graph.get(slug) or [])

    def get_neighborhood(self, slug: str, depth: int = 1) -> dict[str, list[str]]:
        """BFS the crossref graph from `slug` up to `depth` hops in both directions.

        Returns ``{depth_level: [slugs]}`` excluding the starting slug.
        """
        graph = self.compiled.get("graph") or {}
        seen: set[str] = {slug}
        frontier: set[str] = {slug}
        out: dict[str, list[str]] = {}
        for level in range(1, depth + 1):
            next_frontier: set[str] = set()
            for s in frontier:
                next_frontier.update(graph.get(s) or [])
                # backlinks
                for src, refs in graph.items():
                    if s in refs:
                        next_frontier.add(src)
            next_frontier -= seen
            if not next_frontier:
                break
            out[str(level)] = sorted(next_frontier)
            seen.update(next_frontier)
            frontier = next_frontier
        return out
