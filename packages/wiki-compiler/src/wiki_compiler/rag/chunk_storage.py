"""Pinecone-backed RAG dedup for generated chunks (optional, opt-in via env var).

Behavior:
- If PINECONE_API_KEY is set: chunks are saved with metadata + embedding to a
  cross-tenant index. Before saving a new chunk we cosine-search and skip if
  any neighbor exceeds the dedup threshold (default 0.92).
- If PINECONE_API_KEY is NOT set: all methods are graceful no-ops. The compile
  still runs end-to-end; just no cross-tenant learning. This keeps the MVP
  lightweight and lets users opt into the network effect when ready.

The index is shared across all wikis (`wiki-embedded-chunks`) with a
namespace per `thesis_hash`, so different wikis stay isolated unless cross-
tenant similarity is explicitly enabled at query time (passes `namespace=None`).
"""
from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


# Reuse the constant from the polars-data lineage where possible.
INDEX_NAME = "wiki-embedded-chunks"
DEDUP_SIMILARITY_THRESHOLD = 0.92  # cosine — chunks above are considered duplicates


@dataclass
class SimilarChunk:
    id: str
    score: float
    archetype_id: str
    source_query: str
    chunk_body: str
    quality_score: float
    reuse_count: int


def _has_pinecone_credentials() -> bool:
    return bool(os.getenv("PINECONE_API_KEY"))


class ChunkStorage:
    """Optional Pinecone RAG dedup for generated chunks.

    Construct without Pinecone API key → all methods are no-ops.
    Construct with key → save + find_similar are live.
    """

    def __init__(self, embedding_dims: int = 768, cross_tenant: bool = True):
        self.enabled = _has_pinecone_credentials()
        self.embedding_dims = embedding_dims
        self.cross_tenant = cross_tenant
        self._index = None
        if self.enabled:
            self._index = self._init_index()

    def _init_index(self):
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing = {idx.name for idx in pc.list_indexes()}
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=self.embedding_dims,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return pc.Index(INDEX_NAME)

    def find_similar(
        self,
        query_vector: np.ndarray,
        thesis_hash: str,
        top_k: int = 3,
    ) -> list[SimilarChunk]:
        """Return chunks similar to the query vector. Cross-tenant if configured."""
        if not self.enabled:
            return []
        namespace = None if self.cross_tenant else thesis_hash
        resp = self._index.query(
            vector=query_vector.astype(np.float32).tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
        out: list[SimilarChunk] = []
        for m in resp.matches:
            md = m.metadata or {}
            out.append(
                SimilarChunk(
                    id=m.id,
                    score=float(m.score),
                    archetype_id=str(md.get("archetype_id", "")),
                    source_query=str(md.get("source_query", "")),
                    chunk_body=str(md.get("chunk_body", "")),
                    quality_score=float(md.get("quality_score", 0.0)),
                    reuse_count=int(md.get("reuse_count", 0)),
                )
            )
        return out

    def is_duplicate(
        self,
        query_vector: np.ndarray,
        thesis_hash: str,
        threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    ) -> SimilarChunk | None:
        """Return the closest neighbor above `threshold`, or None."""
        if not self.enabled:
            return None
        neighbors = self.find_similar(query_vector, thesis_hash, top_k=1)
        if not neighbors:
            return None
        top = neighbors[0]
        return top if top.score >= threshold else None

    def save_chunk(
        self,
        chunk_id: str,
        vector: np.ndarray,
        thesis_hash: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a chunk + its embedding + metadata. No-op if disabled."""
        if not self.enabled:
            return
        full_metadata = {
            **metadata,
            "thesis_hash": thesis_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reuse_count": int(metadata.get("reuse_count", 0)),
        }
        self._index.upsert(
            vectors=[(chunk_id, vector.astype(np.float32).tolist(), full_metadata)],
            namespace=thesis_hash,
        )

    def increment_reuse_count(self, chunk_id: str, thesis_hash: str) -> None:
        """Atomic-ish bump of reuse_count for survivor-bias scoring."""
        if not self.enabled:
            return
        fetched = self._index.fetch(ids=[chunk_id], namespace=thesis_hash)
        v = fetched.vectors.get(chunk_id)
        if not v:
            return
        new_md = dict(v.metadata)
        new_md["reuse_count"] = int(new_md.get("reuse_count", 0)) + 1
        new_md["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
        self._index.update(id=chunk_id, set_metadata=new_md, namespace=thesis_hash)


def new_chunk_id() -> str:
    return str(uuid.uuid4())
