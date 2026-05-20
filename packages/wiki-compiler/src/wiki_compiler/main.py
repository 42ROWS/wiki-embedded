"""42rows Wiki Compiler — Apify actor entry point.

Pipeline (full compile):
    fetch source -> compile (parse MD + crossref)
                 -> derive thesis (LLM premium one-shot)
                 -> generate query archetypes (LLM)
                 -> embed pages (E5)
                 -> for each archetype:
                        retrieve evidence (cosine)
                        generate chunk (LLM teacher)
                        optional Pinecone dedup
                 -> embed chunks (E5)
                 -> export compiled_wiki.zip + MCP manifest + state.json
                 -> push to KV store + dataset

Incremental update (when `priorBundleUrl`/`priorBundlePath` is supplied):
    the prior bundle is the diff baseline (a "lockfile"). We re-embed only
    added/changed pages, regenerate only chunks whose evidence moved, and reuse
    everything else bit-identically. Falls back to a full rebuild when the prior
    bundle predates the state.json feature, the embedding model changed, or too
    much of the wiki changed.

Pricing events:
- wiki_compile_byok      → user provided LLM key
- wiki_compile_hosted    → 42rows hosted Gemini quota
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from wiki_compiler import incremental as inc
from wiki_compiler.rag.chunk_storage import ChunkStorage, new_chunk_id
from wiki_compiler.state import build_state, content_hash
from wiki_compiler.tenant_store import TenantStore, build_metadata, sync_delta
from wiki_compiler.wiki.chunk_generator import (
    ARCHETYPE_CATEGORIES,
    ChunkPage,
    generate_chunk,
    generate_query_archetypes,
    get_token_usage,
    make_archetype_id,
    reset_token_usage,
    retrieve_evidence,
)
from wiki_compiler.wiki.compiler import CompiledWiki, compile_wiki, sample_pages_for_thesis
from wiki_compiler.wiki.embedder import E5Embedder
from wiki_compiler.wiki.exporter import build_manifest, export_compiled_wiki
from wiki_compiler.wiki.fetch import FetchedWiki, fetch_github, fetch_zip_url, stage_uploaded_files
from wiki_compiler.wiki.thesis_builder import Thesis, derive_thesis

# Default LLM models per provider (premium for thesis, cheap for chunks)
THESIS_MODELS = {"google": "gemini-2.5-pro", "anthropic": "claude-sonnet-4-20250514"}
CHUNK_MODELS = {"google": "gemini-2.5-flash", "anthropic": "claude-haiku-4-5-20251001"}

# New-coverage: archetypes generated per category to cover added pages on an
# incremental update (kept small — targeted at the new content only).
INCREMENTAL_NEW_CHUNKS_PER_CAT = 2

EVIDENCE_TOP_K = 8


# ---------------------------------------------------------------------------
# Apify SDK adapter (graceful fallback for local dev)
# ---------------------------------------------------------------------------

class ApifyAdapter:
    """Thin wrapper around `apify.Actor` with a local-dev fallback.

    Lets the actor run via `python -m src.main` against a local input.json
    without the Apify SDK / platform.
    """

    def __init__(self):
        try:
            from apify import Actor  # type: ignore
            self._Actor = Actor
            self._on_platform = True
        except ImportError:
            self._Actor = None
            self._on_platform = False

    async def __aenter__(self):
        if self._on_platform:
            await self._Actor.init()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._on_platform:
            if exc:
                await self._Actor.fail(status_message=str(exc))
            else:
                await self._Actor.exit()

    async def get_input(self) -> dict[str, Any]:
        if self._on_platform:
            return await self._Actor.get_input() or {}
        # Local dev: read from docker/data/input.json
        local = Path("docker/data/input.json")
        if local.exists():
            return json.loads(local.read_text())
        return {}

    async def set_value(self, key: str, value, content_type: str | None = None) -> str:
        if self._on_platform:
            await self._Actor.set_value(key, value, content_type=content_type)
            store = await self._Actor.open_key_value_store()
            getter = getattr(store, "get_public_url", None)
            if getter is None:
                return key
            url = getter(key)
            # apify-sdk-python 3.x makes this async; older returns a str directly
            if hasattr(url, "__await__"):
                url = await url
            return url or key
        # Local: dump to ./output/
        out = Path("output")
        out.mkdir(exist_ok=True)
        path = out / key
        if isinstance(value, (bytes, bytearray)):
            path.write_bytes(value)
        else:
            path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2))
        return str(path)

    async def push_data(self, item: dict[str, Any]) -> None:
        if self._on_platform:
            await self._Actor.push_data(item)
        else:
            print(f"[push_data] {item.get('kind')}: {item.get('slug', item.get('archetype_id', '?'))}")

    async def charge(self, event_name: str, count: int = 1) -> None:
        if self._on_platform:
            try:
                await self._Actor.charge(event_name=event_name, count=count)
            except Exception as e:  # pricing may not be configured yet
                print(f"[charge] {event_name} x{count} skipped: {e}")
        else:
            print(f"[charge-local] {event_name} x{count}")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _fetch_source(input_data: dict[str, Any]) -> FetchedWiki:
    kind = input_data.get("wikiSource", "github")
    if kind == "github":
        repo = input_data.get("githubRepo")
        if not repo:
            raise ValueError("githubRepo is required when wikiSource=github")
        branch = input_data.get("githubBranch") or "main"
        return fetch_github(repo, branch)
    if kind == "zipUrl":
        url = input_data.get("zipUrl")
        if not url:
            raise ValueError("zipUrl is required when wikiSource=zipUrl")
        return fetch_zip_url(url)
    if kind == "uploadedFiles":
        files = input_data.get("uploadedFiles") or []
        return stage_uploaded_files([Path(f) for f in files])
    raise ValueError(f"Unsupported wikiSource: {kind}")


def _resolve_provider_and_key(input_data: dict[str, Any]) -> tuple[str, str]:
    """Return (provider, api_key). Hosted mode reads our env key."""
    provider = input_data.get("llmProvider", "google")
    if provider == "google":
        # Vertex AI mode uses the service account (ADC) — no api_key needed,
        # and works where the AI Studio API is geo-restricted.
        from wiki_compiler.wiki._google import use_vertex
        if use_vertex():
            return "google", ""
        key = input_data.get("googleApiKey") or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("googleApiKey is required when llmProvider=google (BYOK)")
        return "google", key
    if provider == "anthropic":
        key = input_data.get("anthropicApiKey") or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("anthropicApiKey is required when llmProvider=anthropic (BYOK)")
        return "anthropic", key
    if provider == "hosted":
        key = os.getenv("GEMINI_HOSTED_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_HOSTED_API_KEY env var not set (hosted mode unavailable)")
        return "google", key
    raise ValueError(f"Unsupported llmProvider: {provider}")


def _fetch_prior_bundle(input_data: dict[str, Any]) -> bytes | None:
    """Load the prior compiled bundle (the diff baseline) if the user supplied one.

    `priorBundleUrl` (public URL, e.g. the previous run's compiled_wiki_url) is
    downloaded; `priorBundlePath` (local dev) is read from disk.
    """
    url = input_data.get("priorBundleUrl")
    if url:
        import httpx
        resp = httpx.get(url, timeout=120.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    path = input_data.get("priorBundlePath")
    if path and Path(path).exists():
        return Path(path).read_bytes()
    return None


def _make_query_embedder(embedder: E5Embedder):
    """Return a callable(query: str) -> 1D normalized numpy vector."""
    return embedder.embed_query


# ---------------------------------------------------------------------------
# Compile result (shared shape for full + incremental)
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    thesis_md: str
    thesis_hash: str
    thesis_summary: str
    thesis_derived: bool
    pages: list[dict[str, Any]]
    chunk_dicts: list[dict[str, Any]]
    all_vectors: np.ndarray
    all_slugs: list[str]
    archetypes: list[dict[str, str]]      # [{category, query}] — cached in state
    embedding_model: str
    embedding_dims: int
    teacher_model: str
    categories: list[str]
    crossref_edges: int
    dedup_skipped: int = 0
    mode: str = "full"
    diff_report: dict[str, Any] = field(default_factory=dict)
    # Pinecone sync plan (which slugs to upsert/delete in the tenant namespace).
    upsert_slugs: set[str] = field(default_factory=set)
    delete_ids: list[str] = field(default_factory=list)
    reset_namespace: bool = False

    @property
    def pages_count(self) -> int:
        return len([p for p in self.pages if (p.get("content_full") or "").strip()])

    @property
    def chunks_count(self) -> int:
        return len(self.chunk_dicts)


# ---------------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------------

def _build_full(
    provider: str,
    api_key: str,
    wiki: CompiledWiki,
    *,
    intent: str,
    answer_style: str,
    categories: list[str],
    chunks_per_cat: int,
    embedding_model: str,
    rag_dedup: bool,
    warnings: list[str],
) -> CompileResult:
    # Derive thesis (premium one-shot)
    sample = sample_pages_for_thesis(wiki, n=50)
    print(f"[thesis] deriving from {len(sample)} sample pages via {provider}/{THESIS_MODELS[provider]}")
    thesis: Thesis = derive_thesis(
        provider=provider,  # type: ignore[arg-type]
        intent=intent,
        answer_style=answer_style,
        sample_pages=sample,
        api_key=api_key,
    )
    print(f"[thesis] hash={thesis.hash} summary={thesis.summary[:120]}")

    # Embed pages
    embedder = E5Embedder(model_name=embedding_model)
    page_emb = embedder.embed_pages(wiki.pages, show_progress=False)
    print(f"[embed] {page_emb.vectors.shape[0]} pages embedded ({page_emb.dims}d)")
    query_embedder = _make_query_embedder(embedder)

    # Generate query archetypes
    print(f"[archetypes] generating {chunks_per_cat}x{len(categories)} archetypes")
    archetypes = generate_query_archetypes(
        provider=provider,  # type: ignore[arg-type]
        api_key=api_key,
        model=CHUNK_MODELS[provider],
        pages=wiki.pages,
        thesis_summary=thesis.summary,
        target_audience=thesis.target_audience,
        primary_use_case=thesis.primary_use_case,
        categories=categories,
        chunks_per_category=chunks_per_cat,
    )
    print(f"[archetypes] {len(archetypes)} queries planned")

    # Generate chunks (one LLM call per archetype)
    storage = ChunkStorage(embedding_dims=page_emb.dims) if rag_dedup else ChunkStorage.__new__(ChunkStorage)
    if not rag_dedup:
        storage.enabled = False
    dedup_skipped = 0
    chunk_pages: list[ChunkPage] = []
    for a in archetypes:
        q, cat = a["query"], a["category"]
        evidence = retrieve_evidence(
            q, query_embedder, wiki.pages, page_emb.vectors, page_emb.slug_order, top_k=EVIDENCE_TOP_K
        )
        if not evidence:
            warnings.append(f"no evidence for query: {q[:80]}")
            continue

        # Optional dedup against existing chunks (cross-tenant)
        if storage.enabled:
            qv = query_embedder(q)
            dup = storage.is_duplicate(qv, thesis.hash)
            if dup:
                storage.increment_reuse_count(dup.id, thesis.hash)
                dedup_skipped += 1
                continue

        chunk = generate_chunk(
            query=q,
            category=cat,
            evidence_slugs=evidence,
            pages=wiki.pages,
            slug_set=wiki.slug_set,
            provider=provider,  # type: ignore[arg-type]
            api_key=api_key,
            model=CHUNK_MODELS[provider],
            thesis_summary=thesis.summary,
            target_audience=thesis.target_audience,
            answer_style=thesis.answer_style or answer_style,
            thesis_hash=thesis.hash,
        )
        if chunk is None:
            warnings.append(f"chunk generation failed for: {q[:80]}")
            continue
        chunk_pages.append(chunk)

        # Save to Pinecone for cross-tenant learning
        if storage.enabled:
            cv = query_embedder(q)  # store under the question, not the body
            storage.save_chunk(
                chunk_id=new_chunk_id(),
                vector=cv,
                thesis_hash=thesis.hash,
                metadata={
                    "archetype_id": chunk.archetype_id,
                    "category": chunk.category,
                    "source_query": chunk.source_query,
                    "chunk_body": chunk.body[:4000],
                    "quality_score": chunk.quality_score,
                },
            )

    print(f"[chunks] {len(chunk_pages)} generated, {dedup_skipped} deduped")

    # Embed chunks
    chunk_dicts = [c.to_wiki_page() for c in chunk_pages]
    chunk_emb = embedder.embed_pages(chunk_dicts, show_progress=False)
    print(f"[embed] {chunk_emb.vectors.shape[0]} chunks embedded")

    # Concatenate embeddings (pages first, chunks after)
    if chunk_emb.vectors.shape[0]:
        all_vectors = np.vstack([page_emb.vectors, chunk_emb.vectors])
        all_slugs = list(page_emb.slug_order) + list(chunk_emb.slug_order)
    else:
        all_vectors = page_emb.vectors
        all_slugs = list(page_emb.slug_order)

    return CompileResult(
        thesis_md=thesis.to_markdown(),
        thesis_hash=thesis.hash,
        thesis_summary=thesis.summary,
        thesis_derived=True,
        pages=wiki.pages,
        chunk_dicts=chunk_dicts,
        all_vectors=all_vectors,
        all_slugs=all_slugs,
        archetypes=[{"category": a["category"], "query": a["query"]} for a in archetypes],
        embedding_model=page_emb.model_name,
        embedding_dims=page_emb.dims,
        teacher_model=CHUNK_MODELS[provider],
        categories=categories,
        crossref_edges=wiki.crossref_edges,
        dedup_skipped=dedup_skipped,
        mode="full",
        upsert_slugs=set(all_slugs),
        delete_ids=[],
        reset_namespace=True,  # full rebuild → clear the namespace then upsert all
    )


# ---------------------------------------------------------------------------
# Incremental build
# ---------------------------------------------------------------------------

def _build_incremental(
    provider: str,
    api_key: str,
    wiki: CompiledWiki,
    prior: inc.PriorBundle,
    *,
    embedding_model: str,
    categories: list[str],
    answer_style: str,
    warnings: list[str],
) -> CompileResult | None:
    """Recompute only the delta. Returns None to signal "fall back to full"
    (too much changed, or thesis would need re-derivation)."""
    page_diff = inc.diff_pages(wiki.pages, prior.state.pages)
    frac = page_diff.changed_fraction(len(prior.state.pages))
    if frac > inc.THESIS_REBUILD_THRESHOLD:
        print(f"[incremental] changed fraction {frac:.2f} > {inc.THESIS_REBUILD_THRESHOLD} → full rebuild")
        return None

    thesis_hash = prior.state.thesis_hash
    embedder = E5Embedder(model_name=embedding_model)
    query_embedder = _make_query_embedder(embedder)
    prior_full_map = inc.vecmap_from_result(prior.vectors, prior.slug_order)

    # --- Pages: embed only added/changed, reuse the rest ---
    to_embed = inc.pages_needing_embed(wiki.pages, page_diff, prior)
    fresh_page = embedder.embed_pages(to_embed, show_progress=False) if to_embed else None
    fresh_page_map = (
        inc.vecmap_from_result(fresh_page.vectors, fresh_page.slug_order) if fresh_page is not None else {}
    )
    page_order = [p["slug"] for p in wiki.pages if (p.get("content_full") or "").strip()]
    page_vectors, page_slugs = inc.stack_vectors(page_order, fresh_page_map, prior_full_map)
    print(f"[incremental] pages: +{len(page_diff.added)} ~{len(page_diff.changed)} "
          f"={len(page_diff.unchanged)} -{len(page_diff.removed)} (embedded {len(to_embed)})")

    def retrieve_fn(query: str) -> list[str]:
        if not query or page_vectors.shape[0] == 0:
            return []
        return retrieve_evidence(query, query_embedder, wiki.pages, page_vectors, page_slugs, top_k=EVIDENCE_TOP_K)

    # --- Chunks: classify stale vs carried ---
    stale = inc.stale_chunk_ids(prior.state, page_diff, retrieve_fn)
    aid_meta = {a["id"]: a for a in prior.state.archetypes}  # id -> {id, category, query}

    carried_dicts: list[dict[str, Any]] = []
    for aid in prior.state.chunks:
        if aid in stale:
            continue
        md = prior.chunk_md.get(aid)
        if md is None:
            continue
        carried_dicts.append(inc.parse_chunk_md(md))

    # Stale chunks whose evidence was entirely removed are dropped (not regenerated).
    dropped: list[str] = []
    to_regen: list[str] = []
    for aid in stale:
        ev = set(prior.state.chunks[aid].get("evidence_slugs") or [])
        if ev and ev <= page_diff.removed:
            dropped.append(aid)
        else:
            to_regen.append(aid)

    regen_dicts: list[dict[str, Any]] = []
    for aid in to_regen:
        meta = aid_meta.get(aid)
        query = (meta or {}).get("query") or prior.state.chunks[aid].get("source_query") or ""
        category = (meta or {}).get("category") or (aid.split("--", 1)[0] if "--" in aid else "explanatory")
        if not query:
            continue
        ev = retrieve_fn(query)
        if not ev:
            continue
        chunk = generate_chunk(
            query=query, category=category, evidence_slugs=ev,
            pages=wiki.pages, slug_set=wiki.slug_set,
            provider=provider, api_key=api_key, model=CHUNK_MODELS[provider],  # type: ignore[arg-type]
            thesis_summary=prior.thesis_summary, target_audience="",
            answer_style=answer_style, thesis_hash=thesis_hash,
        )
        if chunk is not None:
            regen_dicts.append(chunk.to_wiki_page())

    # --- New-coverage archetypes seeded from added pages only ---
    new_archetypes: list[dict[str, str]] = []
    new_dicts: list[dict[str, Any]] = []
    added_pages = [p for p in wiki.pages if p["slug"] in page_diff.added]
    if added_pages:
        try:
            new_archetypes = generate_query_archetypes(
                provider=provider, api_key=api_key, model=CHUNK_MODELS[provider],  # type: ignore[arg-type]
                pages=added_pages, thesis_summary=prior.thesis_summary,
                target_audience="", primary_use_case="",
                categories=categories, chunks_per_category=INCREMENTAL_NEW_CHUNKS_PER_CAT,
            )
        except Exception as e:
            warnings.append(f"new-coverage archetype generation failed: {e}")
        existing_ids = set(prior.state.chunks) | {make_archetype_id(a["category"], a["query"]) for a in []}
        for a in new_archetypes:
            aid = make_archetype_id(a["category"], a["query"])
            if aid in existing_ids:
                continue
            ev = retrieve_fn(a["query"])
            if not ev:
                continue
            chunk = generate_chunk(
                query=a["query"], category=a["category"], evidence_slugs=ev,
                pages=wiki.pages, slug_set=wiki.slug_set,
                provider=provider, api_key=api_key, model=CHUNK_MODELS[provider],  # type: ignore[arg-type]
                thesis_summary=prior.thesis_summary, target_audience="",
                answer_style=answer_style, thesis_hash=thesis_hash,
            )
            if chunk is not None:
                new_dicts.append(chunk.to_wiki_page())
                existing_ids.add(aid)

    # --- Embed regenerated + new chunks; reuse carried chunk vectors ---
    regen_plus_new = regen_dicts + new_dicts
    fresh_chunk = embedder.embed_pages(regen_plus_new, show_progress=False) if regen_plus_new else None
    fresh_chunk_map = (
        inc.vecmap_from_result(fresh_chunk.vectors, fresh_chunk.slug_order) if fresh_chunk is not None else {}
    )

    all_chunk_dicts = carried_dicts + regen_plus_new
    chunk_order = [c["slug"] for c in all_chunk_dicts]
    chunk_vectors, chunk_slugs = inc.stack_vectors(chunk_order, fresh_chunk_map, prior_full_map)
    # Keep chunk dicts aligned with the vectors we actually kept.
    kept = set(chunk_slugs)
    all_chunk_dicts = [c for c in all_chunk_dicts if c["slug"] in kept]

    if chunk_vectors.shape[0]:
        all_vectors = np.vstack([page_vectors, chunk_vectors])
        all_slugs = list(page_slugs) + list(chunk_slugs)
    else:
        all_vectors = page_vectors
        all_slugs = list(page_slugs)

    # Archetype cache = prior set + newly added (dedup by id)
    archetypes_used: list[dict[str, str]] = [
        {"category": a["category"], "query": a["query"]} for a in prior.state.archetypes
    ]
    seen_ids = {a["id"] for a in prior.state.archetypes}
    for a in new_archetypes:
        aid = make_archetype_id(a["category"], a["query"])
        if aid not in seen_ids:
            archetypes_used.append({"category": a["category"], "query": a["query"]})
            seen_ids.add(aid)

    diff_report = {
        "pages_added": len(page_diff.added),
        "pages_changed": len(page_diff.changed),
        "pages_unchanged": len(page_diff.unchanged),
        "pages_removed": len(page_diff.removed),
        "pages_embedded": len(to_embed),
        "chunks_carried": len(carried_dicts),
        "chunks_regenerated": len(regen_dicts),
        "chunks_new": len(new_dicts),
        "chunks_dropped": len(dropped),
        "changed_fraction": round(frac, 4),
        "thesis_reused": True,
    }
    print(f"[incremental] chunks: carried={len(carried_dicts)} regen={len(regen_dicts)} "
          f"new={len(new_dicts)} dropped={len(dropped)}")

    return CompileResult(
        thesis_md=prior.thesis_md,
        thesis_hash=thesis_hash,
        thesis_summary=prior.thesis_summary,
        thesis_derived=False,
        pages=wiki.pages,
        chunk_dicts=all_chunk_dicts,
        all_vectors=all_vectors,
        all_slugs=all_slugs,
        archetypes=archetypes_used,
        embedding_model=embedding_model,
        embedding_dims=prior.state.embedding_dims or (int(all_vectors.shape[1]) if all_vectors.shape[0] else 0),
        teacher_model=CHUNK_MODELS[provider],
        categories=categories,
        crossref_edges=wiki.crossref_edges,
        dedup_skipped=0,
        mode="incremental",
        diff_report=diff_report,
        upsert_slugs=(
            page_diff.added | page_diff.changed
            | {c["slug"] for c in regen_dicts} | {c["slug"] for c in new_dicts}
        ),
        delete_ids=(
            list(page_diff.removed) + [f"chunks/{thesis_hash}/{aid}" for aid in dropped]
        ),
        reset_namespace=False,
    )


# ---------------------------------------------------------------------------
# Finalize (export + upload + dataset + output) — shared by both paths
# ---------------------------------------------------------------------------

async def _finalize(
    adapter: ApifyAdapter,
    result: CompileResult,
    *,
    started: float,
    input_data: dict[str, Any],
    source_kind: str,
    source_origin: str,
    answer_style: str,
    emit_mcp_manifest: bool,
    tenant: str | None,
    collection: str,
    warnings: list[str],
) -> dict[str, Any]:
    manifest = build_manifest(
        pages_count=result.pages_count,
        chunks_count=result.chunks_count,
        crossref_edges=result.crossref_edges,
        thesis_hash=result.thesis_hash,
        thesis_summary=result.thesis_summary,
        embedding_model=result.embedding_model,
        embedding_dims=result.embedding_dims,
        source_kind=source_kind,
        source_origin=source_origin,
        answer_style=answer_style,
        chunk_archetypes=result.categories,
        teacher_model=result.teacher_model,
    )

    state = build_state(
        embedding_model=result.embedding_model,
        embedding_dims=result.embedding_dims,
        thesis_hash=result.thesis_hash,
        pages=result.pages,
        chunk_dicts=result.chunk_dicts,
        archetypes=result.archetypes,
    )

    artifact = export_compiled_wiki(
        pages=result.pages,
        chunk_pages=result.chunk_dicts,
        thesis_md=result.thesis_md,
        thesis_hash=result.thesis_hash,
        embedding_vectors=result.all_vectors,
        embedding_slug_order=result.all_slugs,
        manifest=manifest,
        state_json=state.to_json(),
    )

    compiled_url = await adapter.set_value(
        "compiled_wiki.zip", artifact.zip_bytes, content_type="application/zip"
    )
    mcp_url = ""
    if emit_mcp_manifest:
        mcp_url = await adapter.set_value(
            "mcp_manifest.json", manifest, content_type="application/json"
        )

    # Dataset rows (UI dataset view) — read uniformly from chunk dicts.
    for c in result.chunk_dicts:
        fm = c.get("frontmatter") or {}
        cites = list(fm.get("cites") or [])
        await adapter.push_data({
            "kind": "chunk",
            "slug": c.get("slug"),
            "title": c.get("title"),
            "archetype_id": fm.get("archetype_id"),
            "category": fm.get("category"),
            "source_query": fm.get("source_query"),
            "quality_score": fm.get("quality_score"),
            "cites_count": len(cites),
            "cites": cites,
        })

    # Sync the delta to the per-(collection, tenant) Pinecone index (opt-in via tenantId).
    pinecone_result: dict[str, Any] = {"enabled": False}
    if tenant:
        try:
            store = TenantStore(collection, tenant, embedding_dims=result.embedding_dims)
            vec_by_slug = {s: result.all_vectors[i] for i, s in enumerate(result.all_slugs)}
            meta_by_slug = {
                it["slug"]: build_metadata(it, content_hash(it.get("content_full") or ""))
                for it in (result.pages + result.chunk_dicts)
                if it.get("slug")
            }
            pinecone_result = sync_delta(
                store,
                vec_by_slug=vec_by_slug,
                meta_by_slug=meta_by_slug,
                upsert_slugs=result.upsert_slugs,
                delete_ids=result.delete_ids,
                reset=result.reset_namespace,
            )
            print(f"[pinecone] {pinecone_result}")
        except Exception as e:
            warnings.append(f"pinecone sync failed: {e}")
            pinecone_result = {"enabled": True, "error": str(e)}

    event_name = "wiki_compile_byok" if input_data.get("llmProvider") != "hosted" else "wiki_compile_hosted"
    await adapter.charge(event_name=event_name, count=1)

    quality_scores = [
        float((c.get("frontmatter") or {}).get("quality_score", 0.0) or 0.0)
        for c in result.chunk_dicts
    ]
    quality_avg = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    elapsed_ms = int((time.time() - started) * 1000)
    tokens = get_token_usage()
    output = {
        "status": "succeeded",
        "mode": result.mode,
        "wiki_source_kind": source_kind,
        "pages_count": result.pages_count,
        "chunks_count": result.chunks_count,
        "crossref_edges": result.crossref_edges,
        "thesis_hash": result.thesis_hash,
        "thesis_summary": result.thesis_summary,
        "compiled_wiki_url": compiled_url,
        "mcp_manifest_url": mcp_url,
        "embedding_model": result.embedding_model,
        "embedding_dims": result.embedding_dims,
        "llm_calls_count": tokens["calls"] + (1 if result.thesis_derived else 0),
        "llm_tokens_input": tokens["input"],
        "llm_tokens_output": tokens["output"],
        "rag_dedup_skipped": result.dedup_skipped,
        "incremental_diff": result.diff_report,
        "pinecone": pinecone_result,
        "quality_score_avg": round(quality_avg, 2),
        "total_cost_usd": 0.0,
        "execution_time_ms": elapsed_ms,
        "warnings": warnings[:20],
        "errors": "",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    await adapter.set_value("OUTPUT", output, content_type="application/json")
    print(f"[done] {json.dumps({k: v for k, v in output.items() if k != 'warnings'}, indent=2)}")
    return output


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run() -> None:
    started = time.time()
    warnings: list[str] = []

    async with ApifyAdapter() as adapter:
        input_data = await adapter.get_input()

        try:
            reset_token_usage()
            provider, api_key = _resolve_provider_and_key(input_data)
            intent = input_data.get("thesisIntent") or ""
            if not intent.strip():
                raise ValueError("thesisIntent is required")
            answer_style = input_data.get("answerStyle", "concise_actionable")
            categories = input_data.get("chunkArchetypes") or list(ARCHETYPE_CATEGORIES)
            categories = [c for c in categories if c in ARCHETYPE_CATEGORIES] or list(ARCHETYPE_CATEGORIES)
            chunks_per_cat = int(input_data.get("chunksPerArchetype", 4))
            embedding_model = input_data.get("embeddingModel", "pinecone:multilingual-e5-large")
            rag_dedup = bool(input_data.get("ragDedup", True))
            emit_mcp_manifest = bool(input_data.get("outputMcpManifest", True))
            tenant_id = (input_data.get("tenantId") or "").strip() or None
            collection = (input_data.get("collection") or "company-wiki").strip() or "company-wiki"

            # 1. Fetch source
            fetched = _fetch_source(input_data)
            print(f"[fetch] {fetched.source_kind} -> {fetched.local_dir} ({fetched.origin})")

            # 2. Compile
            wiki: CompiledWiki = compile_wiki(fetched.local_dir)
            print(f"[compile] {wiki.pages_count} pages, {wiki.crossref_edges} crossref edges")
            if wiki.parse_errors:
                warnings.extend(wiki.parse_errors[:10])
            if wiki.pages_count == 0:
                raise ValueError("No markdown pages found in source")

            # 3. Dispatch — incremental if a compatible prior bundle is supplied.
            result: CompileResult | None = None
            prior_bytes = _fetch_prior_bundle(input_data)
            if prior_bytes:
                prior = None
                try:
                    prior = inc.load_prior_bundle(prior_bytes)
                except Exception as e:
                    warnings.append(f"prior bundle unreadable: {e}")
                if prior is None:
                    print("[incremental] no usable prior state.json → full rebuild")
                elif prior.state.embedding_model != embedding_model:
                    warnings.append(
                        f"embedding model changed ({prior.state.embedding_model} → {embedding_model}); full rebuild"
                    )
                    print("[incremental] embedding model mismatch → full rebuild")
                else:
                    result = _build_incremental(
                        provider, api_key, wiki, prior,
                        embedding_model=embedding_model, categories=categories,
                        answer_style=answer_style, warnings=warnings,
                    )

            if result is None:
                result = _build_full(
                    provider, api_key, wiki,
                    intent=intent, answer_style=answer_style, categories=categories,
                    chunks_per_cat=chunks_per_cat, embedding_model=embedding_model,
                    rag_dedup=rag_dedup, warnings=warnings,
                )

            # 4. Finalize (export + upload + dataset + output)
            await _finalize(
                adapter, result, started=started, input_data=input_data,
                source_kind=fetched.source_kind, source_origin=fetched.origin,
                answer_style=answer_style, emit_mcp_manifest=emit_mcp_manifest,
                tenant=tenant_id, collection=collection, warnings=warnings,
            )

        except Exception as e:
            err = {
                "status": "failed",
                "errors": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "execution_time_ms": int((time.time() - started) * 1000),
                "timestamp": datetime.now(UTC).isoformat(),
                "warnings": warnings[:20],
            }
            await adapter.set_value("OUTPUT", err, content_type="application/json")
            print(f"[FAILED] {err['errors']}")
            raise


def cli() -> None:
    """Sync entry point for the `wiki-compile` console script."""
    import asyncio
    asyncio.run(run())


if __name__ == "__main__":
    cli()
