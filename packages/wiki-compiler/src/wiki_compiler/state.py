"""Persistent compile state — the diff baseline embedded in every bundle.

`state.json` is the "record manager": it stores, per page and per chunk, the
content hash + the metadata needed to decide, on the next compile, what is new /
changed / unchanged / removed. It also caches the archetype set and records the
embedding model so an incremental update can guard against a vector-space change.

Without this snapshot it is impossible to tell a new item from a changed one, so
`state.json` is what makes incremental updates possible.

See the project CHANGELOG for the incremental-update design.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

STATE_FILENAME = "state.json"
STATE_FORMAT_VERSION = 1


def content_hash(content_full: str) -> str:
    """SHA-256 of the exact text that gets embedded.

    The embedder embeds ``"passage: " + content_full`` (``title\\n\\nbody``), so the
    hash unit is ``content_full``. A frontmatter-only edit therefore does NOT change
    the hash and does NOT trigger a needless re-embed.
    """
    return hashlib.sha256((content_full or "").encode("utf-8")).hexdigest()


@dataclass
class WikiState:
    """The diff baseline. Serialized as ``state.json`` inside the bundle."""

    embedding_model: str
    embedding_dims: int
    thesis_hash: str
    pages: dict[str, str]                 # slug -> content_hash
    archetypes: list[dict[str, str]]      # [{id, category, query}]
    chunks: dict[str, dict[str, Any]]     # archetype_id -> {hash, evidence_slugs, source_query, quality_score}
    format_version: int = STATE_FORMAT_VERSION

    def to_json(self) -> str:
        return json.dumps(
            {
                "format_version": self.format_version,
                "embedding": {"model": self.embedding_model, "dims": self.embedding_dims},
                "thesis": {"hash": self.thesis_hash},
                "pages": self.pages,
                "archetypes": self.archetypes,
                "chunks": self.chunks,
                "counts": {"pages": len(self.pages), "chunks": len(self.chunks)},
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> WikiState:
        d = json.loads(raw)
        emb = d.get("embedding") or {}
        return cls(
            embedding_model=str(emb.get("model", "")),
            embedding_dims=int(emb.get("dims", 0) or 0),
            thesis_hash=str((d.get("thesis") or {}).get("hash", "")),
            pages=dict(d.get("pages") or {}),
            archetypes=list(d.get("archetypes") or []),
            chunks=dict(d.get("chunks") or {}),
            format_version=int(d.get("format_version", STATE_FORMAT_VERSION)),
        )


def build_state(
    *,
    embedding_model: str,
    embedding_dims: int,
    thesis_hash: str,
    pages: list[dict[str, Any]],
    chunk_dicts: list[dict[str, Any]],
    archetypes: list[dict[str, str]],
) -> WikiState:
    """Assemble a :class:`WikiState` from the compile artifacts.

    ``pages`` and ``chunk_dicts`` are wiki-page dicts (the chunk dict carries its
    metadata in ``frontmatter`` — same shape as ``ChunkPage.to_wiki_page``).
    ``archetypes`` is the query set actually used this run (``[{category, query}]``)
    — it is cached so an incremental update reuses it instead of regenerating a
    non-deterministic new set.
    """
    from wiki_compiler.wiki.chunk_generator import make_archetype_id

    page_hashes = {
        p["slug"]: content_hash(p.get("content_full") or "")
        for p in pages
        if (p.get("content_full") or "").strip()
    }
    arche = [
        {"id": make_archetype_id(a["category"], a["query"]), "category": a["category"], "query": a["query"]}
        for a in archetypes
        if a.get("category") and a.get("query")
    ]
    chunk_state: dict[str, dict[str, Any]] = {}
    for c in chunk_dicts:
        fm = c.get("frontmatter") or {}
        aid = fm.get("archetype_id")
        if not aid:
            continue
        title = c.get("title", "")
        body = c.get("body", "")
        chunk_state[str(aid)] = {
            "hash": content_hash(c.get("content_full") or f"{title}\n\n{body}"),
            "evidence_slugs": list(fm.get("evidence_slugs") or []),
            "source_query": str(fm.get("source_query", "")),
            "quality_score": float(fm.get("quality_score", 0.0) or 0.0),
        }
    return WikiState(
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        thesis_hash=thesis_hash,
        pages=page_hashes,
        archetypes=arche,
        chunks=chunk_state,
    )
