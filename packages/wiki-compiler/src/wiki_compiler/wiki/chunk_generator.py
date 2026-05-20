"""LLM-teacher chunk generation — pre-computed answer chunks.

Pipeline (offline, frontloaded at compile time):
1. Generate strategic query archetypes (5 categories × N variants) given thesis
2. For each archetype: retrieve evidence pages (cosine vs E5 embeddings)
3. LLM teacher synthesizes an "ideal answer chunk" citing the evidence slugs
4. Chunk saved as wiki page kind:chunk with provenance frontmatter

The chunks are then embedded and become first-class retrieval targets at
runtime, delivering teacher-quality answers at embedder speed (30-50ms vs ~2s).

Pattern ported from mario-wiki-v2/mariowiki/chunks.py and adapted for the
multi-provider (google + anthropic) BYOK actor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ARCHETYPE_CATEGORIES = (
    "ranking_top_n",
    "comparative",
    "predictive",
    "explanatory",
    "tactical_recipe",
)


CHUNK_GENERATION_PROMPT = """You are an expert strategist. Produce an "ideal answer chunk" grounded strictly in the wiki evidence.

QUERY:
{query}

THESIS (purpose lens — answer should align with this):
{thesis_summary}
Target audience: {target_audience}
Answer style: {answer_style}

EVIDENCE (top retrieved wiki pages):
{evidence}

Task: produce a 150-350 word answer that:
1. Answers the QUERY directly (no preamble)
2. Cites AT LEAST 2 distinct evidence slugs inline as [[slug]] — non-negotiable
3. Uses ONLY the evidence above (no hallucination)
4. Matches the THESIS style and audience

Output JSON, strict schema:
{{
  "title": "max 10 words",
  "body": "the answer, 150-350 words, with [[slug]] citations inline (≥2 distinct)",
  "cites": ["slug1", "slug2", ...]  // MUST contain ≥2 distinct slugs from the evidence
}}

Output ONLY the JSON object. No preamble, no code fence, no extra text."""


QUERY_ARCHETYPES_PROMPT = """You are a product strategist generating realistic queries for a knowledge wiki.

WIKI CONTEXT (excerpt):
{context}

THESIS:
{thesis_summary}
Target audience: {target_audience}
Primary use case: {primary_use_case}

Task: generate {n_total} strategic query archetypes distributed evenly across these categories:
{categories_desc}

Constraints:
- Realistic queries the target audience would actually ask
- Answerable by reading the wiki
- 1-2 sentences each
- {n_per_cat} queries per category

Output JSON: {{"queries": [{{"category": "ranking_top_n", "query": "..."}}, ...]}}

Output ONLY the JSON object. No preamble, no code fence, no extra text."""


CATEGORY_DESCRIPTIONS = {
    "ranking_top_n": "Top-N rankings with rationale (best X, most important Y, leading Z)",
    "comparative": "Compare 2+ options or entities (X vs Y, which is better when...)",
    "predictive": "Probability / forecast / fit estimation (will X work, expected outcome)",
    "explanatory": "Why does X happen / what differentiates X (causal reasoning)",
    "tactical_recipe": "How-to / step-by-step actions (concrete playbook or recipe)",
}


# Module-level token usage tracker — accumulates across calls within one run.
# main.py resets at start and reads at the end.
_TOKEN_USAGE = {"input": 0, "output": 0, "calls": 0}


def reset_token_usage() -> None:
    _TOKEN_USAGE.update({"input": 0, "output": 0, "calls": 0})


def get_token_usage() -> dict[str, int]:
    return dict(_TOKEN_USAGE)


@dataclass
class ChunkPage:
    """A generated chunk in wiki-page format (ready to compile alongside pages)."""

    slug: str
    title: str
    body: str
    archetype_id: str
    category: str
    source_query: str
    cites: list[str] = field(default_factory=list)
    evidence_slugs: list[str] = field(default_factory=list)
    teacher_model: str = ""
    thesis_hash: str = ""
    quality_score: float = 0.0

    def to_wiki_page(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "body": self.body,
            "content_full": f"{self.title}\n\n{self.body}",
            "frontmatter": {
                "kind": "chunk",
                "archetype_id": self.archetype_id,
                "category": self.category,
                "thesis_hash": self.thesis_hash,
                "source_query": self.source_query,
                "cites": self.cites,
                "evidence_slugs": self.evidence_slugs,
                "teacher": self.teacher_model,
                "quality_score": self.quality_score,
            },
            "crossrefs": set(self.cites),
        }


def _build_context_excerpt(pages: list[dict[str, Any]], max_pages: int = 30, max_chars_per_page: int = 300) -> str:
    parts = []
    for p in pages[:max_pages]:
        title = (p.get("title") or "").strip()
        slug = p.get("slug", "")
        body = (p.get("body") or "")[:max_chars_per_page].strip()
        parts.append(f"## [[{slug}]] — {title}\n{body}")
    return "\n\n".join(parts)


def _format_evidence(pages: list[dict[str, Any]], slugs: list[str], max_chars_per_page: int = 1000) -> str:
    by_slug = {p["slug"]: p for p in pages}
    parts = []
    for s in slugs:
        p = by_slug.get(s)
        if not p:
            continue
        title = (p.get("title") or "").strip()
        body = (p.get("body") or "")[:max_chars_per_page].strip()
        parts.append(f"## [[{s}]] — {title}\n{body}")
    return "\n\n".join(parts)


def retrieve_evidence(
    query: str,
    query_embedder,
    pages: list[dict[str, Any]],
    page_vectors: np.ndarray,
    slug_order: list[str],
    top_k: int = 8,
) -> list[str]:
    """E5 cosine retrieval of top-K evidence slugs for a query.

    `query_embedder` must be a callable producing a unit-normalized
    1D numpy vector for the given query (E5 'query: ' prefix applied inside).
    """
    qv = query_embedder(query)
    scores = page_vectors @ qv  # cosine since both normalized
    top_idx = np.argsort(-scores)[:top_k]
    return [slug_order[i] for i in top_idx if i < len(slug_order)]


# ---------------------------------------------------------------------------
# Provider-specific calls
# ---------------------------------------------------------------------------

def _call_google(prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    from google.genai import types as gtypes

    from wiki_compiler.wiki._google import genai_client
    client = genai_client(api_key)
    cfg = gtypes.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
    )
    r = client.models.generate_content(model=model, contents=prompt, config=cfg)
    usage = getattr(r, "usage_metadata", None)
    if usage is not None:
        _TOKEN_USAGE["input"] += int(getattr(usage, "prompt_token_count", 0) or 0)
        _TOKEN_USAGE["output"] += int(getattr(usage, "candidates_token_count", 0) or 0)
        _TOKEN_USAGE["calls"] += 1
    return r.text or "{}"


def _call_anthropic(prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(msg, "usage", None)
    if usage is not None:
        _TOKEN_USAGE["input"] += int(getattr(usage, "input_tokens", 0) or 0)
        _TOKEN_USAGE["output"] += int(getattr(usage, "output_tokens", 0) or 0)
        _TOKEN_USAGE["calls"] += 1
    return "".join(block.text for block in msg.content if block.type == "text")


def _llm_call(provider: str, prompt: str, api_key: str, model: str, max_tokens: int = 3000) -> str:
    if provider == "google":
        return _call_google(prompt, api_key, model, max_tokens)
    if provider == "anthropic":
        return _call_anthropic(prompt, api_key, model, max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_query_archetypes(
    *,
    provider: Literal["google", "anthropic"],
    api_key: str,
    model: str,
    pages: list[dict[str, Any]],
    thesis_summary: str,
    target_audience: str,
    primary_use_case: str,
    categories: list[str],
    chunks_per_category: int,
) -> list[dict[str, str]]:
    """Generate strategic query archetypes per category. Returns list of {category, query}."""
    n_total = chunks_per_category * len(categories)
    cat_desc = "\n".join(f"- {c}: {CATEGORY_DESCRIPTIONS[c]}" for c in categories)
    prompt = QUERY_ARCHETYPES_PROMPT.format(
        context=_build_context_excerpt(pages),
        thesis_summary=thesis_summary,
        target_audience=target_audience,
        primary_use_case=primary_use_case,
        n_total=n_total,
        n_per_cat=chunks_per_category,
        categories_desc=cat_desc,
    )
    raw = _llm_call(provider, prompt, api_key, model, max_tokens=4000)
    parsed = json.loads(raw)
    queries = parsed.get("queries", [])
    return [
        {"category": str(q["category"]), "query": str(q["query"])}
        for q in queries
        if isinstance(q, dict) and q.get("category") in categories and q.get("query")
    ]


def generate_chunk(
    *,
    query: str,
    category: str,
    evidence_slugs: list[str],
    pages: list[dict[str, Any]],
    slug_set: set[str],
    provider: Literal["google", "anthropic"],
    api_key: str,
    model: str,
    thesis_summary: str,
    target_audience: str,
    answer_style: str,
    thesis_hash: str,
) -> ChunkPage | None:
    """Generate a single answer chunk by asking the LLM to synthesize evidence.

    Returns None if the LLM produced unusable output (no body or no in-corpus cites).
    """
    evidence_text = _format_evidence(pages, evidence_slugs)
    if not evidence_text.strip():
        return None

    prompt = CHUNK_GENERATION_PROMPT.format(
        query=query,
        thesis_summary=thesis_summary,
        target_audience=target_audience,
        answer_style=answer_style,
        evidence=evidence_text,
    )
    raw = _llm_call(provider, prompt, api_key, model, max_tokens=2500)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None

    title = str(parsed.get("title", ""))[:200]
    body = str(parsed.get("body", ""))[:4000]
    cites_raw = parsed.get("cites", [])
    cites = [str(c) for c in cites_raw if isinstance(c, str) and c in slug_set]

    if not body.strip():
        return None

    # Quality score 0-100:
    # - body length within range (150-350 words): 30
    # - has cites in-corpus: 30 (10 per cite, max 3)
    # - title present: 10
    # - all evidence_slugs cited (coverage): 30
    word_count = len(body.split())
    score = 0.0
    if 100 <= word_count <= 450:
        score += 30.0
    elif word_count >= 50:
        score += 15.0
    score += min(30.0, len(cites) * 10.0)
    if title.strip():
        score += 10.0
    if evidence_slugs:
        coverage = len(set(cites) & set(evidence_slugs)) / max(1, len(evidence_slugs))
        score += 30.0 * coverage

    aid = make_archetype_id(category, query)
    slug = f"chunks/{thesis_hash}/{aid}" if thesis_hash else f"chunks/{aid}"

    return ChunkPage(
        slug=slug,
        title=title or query[:80],
        body=body,
        archetype_id=aid,
        category=category,
        source_query=query,
        cites=cites,
        evidence_slugs=evidence_slugs,
        teacher_model=model,
        thesis_hash=thesis_hash,
        quality_score=round(min(score, 100.0), 2),
    )


def _slugify(s: str) -> str:
    import re
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s.strip().lower())
    return s.strip("-")[:60]


def make_archetype_id(category: str, query: str) -> str:
    """Deterministic, stable chunk identity given (category, query).

    Used as the chunk's archetype_id and (with thesis_hash) its slug. Stability
    across compiles is what makes incremental diffing of chunks possible
    (see the project CHANGELOG).
    """
    return f"{category}--{_slugify(query[:60])}"
