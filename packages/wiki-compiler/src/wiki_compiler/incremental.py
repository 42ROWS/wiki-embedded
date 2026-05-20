"""Incremental update — diff a new source against a prior bundle and recompute
only the delta.

The prior bundle (the ``compiled_wiki.zip`` the user got last time) is the diff
baseline, exactly like a lockfile: the user passes it back, we read its
``state.json`` + ``embeddings.npz``, compare, and re-embed / re-generate only what
changed. Page embeddings are reused bit-identically; chunks are regenerated only
when their evidence moved.

Pure logic (stdlib + numpy) — no LLM, no network — so it is unit-testable. The LLM
regeneration and the network fetch of the prior bundle are orchestrated by
``main.py``.

See the project CHANGELOG for the incremental-update design.
"""
from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import frontmatter
import numpy as np

from wiki_compiler.state import STATE_FILENAME, WikiState, content_hash

# Changed-fraction thresholds (see ADR §"Manopole").
THESIS_REBUILD_THRESHOLD = 0.30   # above → re-derive thesis (resets chunk identity)
FULL_REBUILD_FRACTION = 0.70      # above → skip incremental, just full rebuild


# ---------------------------------------------------------------------------
# Prior bundle reader
# ---------------------------------------------------------------------------

@dataclass
class PriorBundle:
    state: WikiState
    vectors: np.ndarray              # (N, dims) float32
    slug_order: list[str]
    vec_by_slug: dict[str, int]      # slug -> row index in `vectors`
    chunk_md: dict[str, str]         # archetype_id -> raw markdown of the chunk file
    thesis_md: str
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def thesis_summary(self) -> str:
        return str((self.manifest.get("thesis") or {}).get("summary", ""))

    def page_vector(self, slug: str) -> np.ndarray | None:
        i = self.vec_by_slug.get(slug)
        return None if i is None else self.vectors[i]

    def chunk_vector(self, archetype_id: str, thesis_hash: str) -> np.ndarray | None:
        return self.page_vector(f"chunks/{thesis_hash}/{archetype_id}")


def load_prior_bundle(zip_bytes: bytes) -> PriorBundle | None:
    """Read a previously compiled bundle. Returns ``None`` if it predates the
    ``state.json`` feature (caller then falls back to a full rebuild)."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = set(zf.namelist())
    if STATE_FILENAME not in names:
        return None

    state = WikiState.from_json(zf.read(STATE_FILENAME))

    npz = np.load(io.BytesIO(zf.read("embeddings.npz")), allow_pickle=True)
    vectors = np.asarray(npz["vectors"], dtype=np.float32)
    slug_order = [str(s) for s in npz["slug_order"].tolist()]
    vec_by_slug = {s: i for i, s in enumerate(slug_order)}

    chunk_md: dict[str, str] = {}
    for name in names:
        if name.startswith("chunks/") and name.endswith(".md"):
            stem = name.rsplit("/", 1)[-1][:-3]   # archetype_id == file stem
            chunk_md[stem] = zf.read(name).decode("utf-8")

    thesis_md = zf.read("thesis.md").decode("utf-8") if "thesis.md" in names else ""
    manifest: dict[str, Any] = {}
    if "manifest.json" in names:
        import json
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except Exception:
            manifest = {}
    return PriorBundle(state, vectors, slug_order, vec_by_slug, chunk_md, thesis_md, manifest)


def parse_chunk_md(md_text: str) -> dict[str, Any]:
    """Reconstruct a chunk wiki-page dict from a carried-over chunk markdown file.

    Mirrors ``ChunkPage.to_wiki_page`` so carried chunks export identically to
    freshly generated ones.
    """
    post = frontmatter.loads(md_text)
    fm = dict(post.metadata)
    body = post.content.strip()
    title = str(fm.get("title", ""))
    cites = list(fm.get("cites") or [])
    return {
        "slug": str(fm.get("slug", "")),
        "title": title,
        "body": body,
        "content_full": f"{title}\n\n{body}",
        "frontmatter": fm,
        "crossrefs": set(cites),
    }


# ---------------------------------------------------------------------------
# Page diff
# ---------------------------------------------------------------------------

@dataclass
class PageDiff:
    added: set[str] = field(default_factory=set)
    changed: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)

    @property
    def touched(self) -> set[str]:
        """Pages whose change can invalidate a chunk (changed + removed)."""
        return self.changed | self.removed

    def changed_fraction(self, prior_page_count: int) -> float:
        if prior_page_count <= 0:
            return 1.0
        return (len(self.added) + len(self.changed) + len(self.removed)) / prior_page_count


def diff_pages(new_pages: list[dict[str, Any]], prior_page_hashes: dict[str, str]) -> PageDiff:
    """Classify each page as added / changed / unchanged / removed by comparing
    ``sha256(content_full)`` against the prior snapshot."""
    new_hashes = {
        p["slug"]: content_hash(p.get("content_full") or "")
        for p in new_pages
        if (p.get("content_full") or "").strip()
    }
    new_slugs = set(new_hashes)
    prior_slugs = set(prior_page_hashes)
    common = new_slugs & prior_slugs
    changed = {s for s in common if new_hashes[s] != prior_page_hashes[s]}
    return PageDiff(
        added=new_slugs - prior_slugs,
        changed=changed,
        unchanged=common - changed,
        removed=prior_slugs - new_slugs,
    )


# ---------------------------------------------------------------------------
# Chunk staleness
# ---------------------------------------------------------------------------

def stale_chunk_ids(
    prior_state: WikiState,
    page_diff: PageDiff,
    retrieve_fn: Callable[[str], list[str]] | None = None,
) -> set[str]:
    """Archetype ids whose chunk must be regenerated.

    A chunk is stale if its stored evidence touched a changed/removed page, OR
    (evidence drift) if re-running retrieval on the new corpus yields a different
    top-k evidence set. ``retrieve_fn(query)`` is cheap (cosine, no LLM); pass
    ``None`` to skip the drift check (evidence-only mode).
    """
    touched = page_diff.touched
    stale: set[str] = set()
    for aid, meta in prior_state.chunks.items():
        ev = set(meta.get("evidence_slugs") or [])
        if ev & touched:
            stale.add(aid)
            continue
        if retrieve_fn is not None:
            new_ev = set(retrieve_fn(meta.get("source_query") or ""))
            if new_ev and new_ev != ev:
                stale.add(aid)
    return stale


# ---------------------------------------------------------------------------
# Vector carry-over
# ---------------------------------------------------------------------------

def stack_vectors(
    order: list[str],
    *vec_maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Build a (M, dims) matrix for ``order``, taking each slug's vector from the
    first ``vec_maps`` that contains it. Slugs absent from all maps are dropped
    (and excluded from the returned order)."""
    rows: list[np.ndarray] = []
    kept: list[str] = []
    for slug in order:
        for m in vec_maps:
            v = m.get(slug)
            if v is not None:
                rows.append(np.asarray(v, dtype=np.float32))
                kept.append(slug)
                break
    if not rows:
        dims = 0
        for m in vec_maps:
            for v in m.values():
                dims = int(np.asarray(v).shape[-1])
                break
            if dims:
                break
        return np.zeros((0, dims), dtype=np.float32), []
    return np.vstack(rows), kept


def vecmap_from_result(vectors: np.ndarray, slug_order: list[str]) -> dict[str, np.ndarray]:
    """Turn parallel (vectors, slug_order) into a slug -> vector map."""
    return {slug: vectors[i] for i, slug in enumerate(slug_order)}


def pages_needing_embed(new_pages: list[dict[str, Any]], page_diff: PageDiff, prior: PriorBundle) -> list[dict[str, Any]]:
    """Pages to embed fresh: added/changed, plus any page missing from the prior
    vectors (defensive — keeps every kept page backed by a vector)."""
    return [
        p
        for p in new_pages
        if (p.get("content_full") or "").strip()
        and (p["slug"] in page_diff.changed
             or p["slug"] in page_diff.added
             or p["slug"] not in prior.vec_by_slug)
    ]
