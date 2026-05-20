"""Per-(collection, tenant) Pinecone store — the live retrieval index.

Pinecone maps cleanly onto our two axes:

    collection  ->  Index       (a corpus type, e.g. "company-wiki", "marketing-skills")
    tenant      ->  Namespace   (one customer; Pinecone's native multi-tenancy)

A :class:`TenantStore` is a handle **bound** to one ``(collection, tenant)`` at
construction. Every ``upsert`` / ``delete`` / ``query`` is confined to that
tenant's namespace inside that collection's index, so isolation is by
construction — a handle can neither read nor write another tenant or collection.

(NB: "collection" here = a Pinecone *index*, not Pinecone's "Collection" feature,
which is a static index backup. Different concept, same word.)

If ``PINECONE_API_KEY`` is absent the store is a graceful no-op (``enabled`` is
False), so a compile still produces its bundle without a live index.

See the project CHANGELOG for the incremental-update design.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# Pinecone metadata caps at ~40KB/vector; keep grounding text well under that.
_MAX_TEXT_CHARS = 8000
_UPSERT_BATCH = 100

_INDEX_NAME_RE = re.compile(r"[^a-z0-9-]+")


def index_name_for(collection: str) -> str:
    """Normalize a collection name into a valid Pinecone index name
    (lowercase, alphanumeric + '-', max 45 chars)."""
    s = _INDEX_NAME_RE.sub("-", collection.strip().lower()).strip("-")
    return (s or "wiki")[:45].strip("-")


@dataclass
class SearchHit:
    id: str
    score: float
    kind: str
    title: str
    text: str
    slug: str
    cites: list[str]


def build_metadata(page: dict[str, Any], content_hash: str) -> dict[str, Any]:
    """Pinecone metadata for a page/chunk: enough to ground on without a 2nd fetch."""
    fm = page.get("frontmatter") or {}
    kind = str(fm.get("kind") or "page")
    if kind == "chunk":
        text = (page.get("body") or "")[:_MAX_TEXT_CHARS]
    else:
        text = (page.get("content_full") or "")[:_MAX_TEXT_CHARS]
    md: dict[str, Any] = {
        "kind": kind,
        "slug": str(page.get("slug", "")),
        "title": str(page.get("title", "")),
        "text": text,
        "content_hash": content_hash,
    }
    if kind == "chunk":
        md["cites"] = [str(c) for c in (fm.get("cites") or [])]
        md["category"] = str(fm.get("category", ""))
        md["quality_score"] = float(fm.get("quality_score", 0.0) or 0.0)
    return md


def _has_pinecone() -> bool:
    return bool(os.getenv("PINECONE_API_KEY"))


class TenantStore:
    """Pinecone handle scoped to one ``(collection, tenant)``.

    All operations are confined to ``namespace == tenant`` inside the
    ``collection`` index. Construct without a Pinecone key → no-op handle.
    """

    def __init__(
        self,
        collection: str,
        tenant: str,
        *,
        embedding_dims: int = 1024,
        create: bool = True,
        index=None,  # injectable for tests
    ):
        if not tenant:
            raise ValueError("tenant (namespace) is required")
        self.collection = collection
        self.index_name = index_name_for(collection)
        self.tenant = tenant  # == namespace
        self.embedding_dims = embedding_dims
        self._index = index
        self.enabled = index is not None or _has_pinecone()
        if self._index is None and self.enabled:
            self._index = self._ensure_index(create)

    @property
    def namespace(self) -> str:
        return self.tenant

    def _ensure_index(self, create: bool):
        import time as _time

        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing = {idx.name for idx in pc.list_indexes()}
        if self.index_name not in existing:
            if not create:
                raise ValueError(f"Pinecone index '{self.index_name}' does not exist")
            pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dims,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # A freshly-created serverless index is not immediately queryable.
            for _ in range(60):
                try:
                    if pc.describe_index(self.index_name).status.get("ready"):
                        break
                except Exception:
                    pass
                _time.sleep(1)
        return pc.Index(self.index_name)

    # ------------------------------------------------------------------ writes
    def upsert(self, records: list[tuple[str, np.ndarray, dict[str, Any]]]) -> int:
        """Upsert (id, vector, metadata) tuples into this tenant's namespace.

        Idempotent by id (Pinecone overwrites same-id), so re-upserting an
        unchanged vector is harmless. Returns the count upserted."""
        if not self.enabled or not records:
            return 0
        n = 0
        for i in range(0, len(records), _UPSERT_BATCH):
            batch = records[i : i + _UPSERT_BATCH]
            self._index.upsert(
                vectors=[
                    (rid, np.asarray(vec, dtype=np.float32).tolist(), md)
                    for rid, vec, md in batch
                ],
                namespace=self.tenant,
            )
            n += len(batch)
        return n

    def delete(self, ids: list[str] | None = None, *, all: bool = False) -> int:
        """Delete by id (or the whole namespace with ``all=True``)."""
        if not self.enabled:
            return 0
        if all:
            try:
                self._index.delete(delete_all=True, namespace=self.tenant)
            except Exception as e:
                # A fresh index has no namespace yet → nothing to clear (404).
                if "not found" not in str(e).lower() and "404" not in str(e):
                    raise
            return -1
        if not ids:
            return 0
        for i in range(0, len(ids), _UPSERT_BATCH):
            self._index.delete(ids=ids[i : i + _UPSERT_BATCH], namespace=self.tenant)
        return len(ids)

    # ------------------------------------------------------------------ reads
    def query(self, vector: np.ndarray, top_k: int = 8) -> list[SearchHit]:
        """Top-k vectors in this tenant's namespace."""
        if not self.enabled:
            return []
        resp = self._index.query(
            vector=np.asarray(vector, dtype=np.float32).tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.tenant,
        )
        out: list[SearchHit] = []
        for m in resp.matches:
            md = m.metadata or {}
            out.append(
                SearchHit(
                    id=m.id,
                    score=float(m.score),
                    kind=str(md.get("kind", "")),
                    title=str(md.get("title", "")),
                    text=str(md.get("text", "")),
                    slug=str(md.get("slug", "")),
                    cites=[str(c) for c in (md.get("cites") or [])],
                )
            )
        return out

    def search(self, query: str, embedder, top_k: int = 8) -> list[SearchHit]:
        """Embed ``query`` (same model as indexing) and return top-k hits."""
        if not self.enabled or not query.strip():
            return []
        qv = embedder.embed_query(query)
        return self.query(qv, top_k=top_k)


def sync_delta(
    store: TenantStore,
    *,
    vec_by_slug: dict[str, np.ndarray],
    meta_by_slug: dict[str, dict[str, Any]],
    upsert_slugs: set[str],
    delete_ids: list[str],
    reset: bool,
) -> dict[str, Any]:
    """Apply a compile delta to the tenant namespace.

    ``reset`` (full rebuild) clears the namespace first, then upserts everything;
    otherwise it is a pure delta (upsert added/changed/regenerated, delete removed).
    Carried/unchanged vectors are simply not in ``upsert_slugs`` — that is the saving.
    """
    if not store.enabled:
        return {"enabled": False}
    if reset:
        store.delete(all=True)
    records = [
        (s, vec_by_slug[s], meta_by_slug[s])
        for s in upsert_slugs
        if s in vec_by_slug and s in meta_by_slug
    ]
    upserted = store.upsert(records)
    deleted = store.delete(ids=delete_ids) if (delete_ids and not reset) else 0
    return {
        "enabled": True,
        "index": store.index_name,
        "namespace": store.namespace,
        "upserted": upserted,
        "deleted": deleted,
        "reset": reset,
    }
