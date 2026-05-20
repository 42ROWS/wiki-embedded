"""Unit tests for the incremental-update logic (no LLM / no embedding model).

Covers the diff baseline (state.json), page diff, chunk staleness, vector
carry-over, and a full export -> reload round-trip.
See ADR 2026-05-20-incremental-wiki-embedding-update.
"""
from __future__ import annotations

import numpy as np

from wiki_compiler import incremental as inc
from wiki_compiler.state import WikiState, build_state, content_hash
from wiki_compiler.wiki.exporter import export_compiled_wiki

# --------------------------------------------------------------------------- helpers

def _page(slug: str, title: str, body: str) -> dict:
    return {
        "slug": slug,
        "title": title,
        "body": body,
        "content_full": f"{title}\n\n{body}",
        "frontmatter": {"slug": slug},
        "crossrefs": set(),
    }


def _chunk_dict(archetype_id: str, category: str, query: str, cites: list[str], thesis_hash: str) -> dict:
    title, body = f"Chunk {archetype_id}", f"Answer citing {cites}."
    return {
        "slug": f"chunks/{thesis_hash}/{archetype_id}",
        "title": title,
        "body": body,
        "content_full": f"{title}\n\n{body}",
        "frontmatter": {
            "kind": "chunk",
            "archetype_id": archetype_id,
            "category": category,
            "thesis_hash": thesis_hash,
            "source_query": query,
            "cites": cites,
            "evidence_slugs": cites,
            "teacher": "test",
            "quality_score": 80.0,
        },
        "crossrefs": set(cites),
    }


# --------------------------------------------------------------------------- content_hash

def test_content_hash_matches_embed_unit_only():
    # Same content_full => same hash regardless of surrounding frontmatter.
    assert content_hash("About\n\nwe sell X") == content_hash("About\n\nwe sell X")
    assert content_hash("About\n\nwe sell X") != content_hash("About\n\nwe sell Y")


# --------------------------------------------------------------------------- state round-trip

def test_state_json_round_trip():
    s = WikiState(
        embedding_model="pinecone:multilingual-e5-large",
        embedding_dims=1024,
        thesis_hash="THX",
        pages={"about": "h1"},
        archetypes=[{"id": "explanatory--why", "category": "explanatory", "query": "why?"}],
        chunks={"explanatory--why": {"hash": "ch", "evidence_slugs": ["about"], "source_query": "why?", "quality_score": 90.0}},
    )
    s2 = WikiState.from_json(s.to_json())
    assert s2.embedding_model == s.embedding_model
    assert s2.embedding_dims == 1024
    assert s2.thesis_hash == "THX"
    assert s2.pages == {"about": "h1"}
    assert s2.chunks["explanatory--why"]["evidence_slugs"] == ["about"]


# --------------------------------------------------------------------------- page diff

def test_diff_pages_classifies_correctly():
    new_pages = [
        _page("about", "About", "unchanged body"),
        _page("pricing", "Pricing", "NEW price"),
        _page("careers", "Careers", "we are hiring"),
    ]
    prior = {
        "about": content_hash("About\n\nunchanged body"),
        "pricing": content_hash("Pricing\n\nOLD price"),
        "gone": content_hash("Gone\n\nremoved"),
    }
    d = inc.diff_pages(new_pages, prior)
    assert d.added == {"careers"}
    assert d.changed == {"pricing"}
    assert d.unchanged == {"about"}
    assert d.removed == {"gone"}
    assert d.touched == {"pricing", "gone"}
    assert abs(d.changed_fraction(3) - (1 + 1 + 1) / 3) < 1e-9


# --------------------------------------------------------------------------- chunk staleness

def test_stale_by_evidence_intersection():
    state = WikiState(
        embedding_model="m", embedding_dims=4, thesis_hash="T",
        pages={}, archetypes=[],
        chunks={
            "c1": {"hash": "x", "evidence_slugs": ["about", "pricing"], "source_query": "q1", "quality_score": 1},
            "c2": {"hash": "y", "evidence_slugs": ["team"], "source_query": "q2", "quality_score": 1},
        },
    )
    diff = inc.PageDiff(changed={"pricing"})
    stale = inc.stale_chunk_ids(state, diff, retrieve_fn=None)
    assert stale == {"c1"}  # c2 untouched, drift check disabled


def test_stale_by_evidence_drift():
    state = WikiState(
        embedding_model="m", embedding_dims=4, thesis_hash="T",
        pages={}, archetypes=[],
        chunks={"c2": {"hash": "y", "evidence_slugs": ["team"], "source_query": "q2", "quality_score": 1}},
    )
    diff = inc.PageDiff()  # nothing touched
    # retrieval now returns a different evidence set => drift => stale
    stale = inc.stale_chunk_ids(state, diff, retrieve_fn=lambda _q: ["team", "newpage"])
    assert stale == {"c2"}
    # same evidence => not stale
    assert inc.stale_chunk_ids(state, diff, retrieve_fn=lambda _q: ["team"]) == set()


# --------------------------------------------------------------------------- vector carry-over

def test_stack_vectors_prefers_fresh_then_prior_drops_missing():
    fresh = {"a": np.array([1, 1], dtype=np.float32), "b": np.array([2, 2], dtype=np.float32)}
    prior = {"a": np.array([9, 9], dtype=np.float32), "c": np.array([3, 3], dtype=np.float32)}
    mat, kept = inc.stack_vectors(["a", "b", "c", "missing"], fresh, prior)
    assert kept == ["a", "b", "c"]
    assert mat.shape == (3, 2)
    assert np.array_equal(mat[0], [1, 1])   # fresh wins over prior for "a"
    assert np.array_equal(mat[1], [2, 2])
    assert np.array_equal(mat[2], [3, 3])   # carried from prior


def test_pages_needing_embed():
    new_pages = [_page("about", "About", "x"), _page("pricing", "Pricing", "y"), _page("careers", "Careers", "z")]
    diff = inc.PageDiff(added={"careers"}, changed={"pricing"}, unchanged={"about"})
    prior = inc.PriorBundle(
        state=WikiState("m", 2, "T", {}, [], {}),
        vectors=np.zeros((1, 2), dtype=np.float32),
        slug_order=["about"],
        vec_by_slug={"about": 0},
        chunk_md={}, thesis_md="",
    )
    need = {p["slug"] for p in inc.pages_needing_embed(new_pages, diff, prior)}
    assert need == {"pricing", "careers"}  # about is unchanged AND present in prior vectors


# --------------------------------------------------------------------------- export -> reload round-trip

def test_export_then_load_prior_bundle_round_trip():
    thesis_hash = "THX"
    pages = [_page("about", "About", "we sell X")]
    chunk = _chunk_dict("ranking_top_n--best", "ranking_top_n", "best?", ["about"], thesis_hash)
    vectors = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    slug_order = ["about", chunk["slug"]]

    state = build_state(
        embedding_model="local-test", embedding_dims=4, thesis_hash=thesis_hash,
        pages=pages, chunk_dicts=[chunk], archetypes=[{"category": "ranking_top_n", "query": "best?"}],
    )
    artifact = export_compiled_wiki(
        pages=pages, chunk_pages=[chunk], thesis_md="# Thesis", thesis_hash=thesis_hash,
        embedding_vectors=vectors, embedding_slug_order=slug_order,
        manifest={"thesis": {"summary": "Sells X"}}, state_json=state.to_json(),
    )

    prior = inc.load_prior_bundle(artifact.zip_bytes)
    assert prior is not None
    assert prior.state.embedding_model == "local-test"
    assert prior.state.embedding_dims == 4
    assert prior.state.thesis_hash == thesis_hash
    assert prior.state.pages["about"] == content_hash("About\n\nwe sell X")
    assert prior.state.chunks["ranking_top_n--best"]["evidence_slugs"] == ["about"]
    assert prior.thesis_summary == "Sells X"

    # vectors recoverable by slug (carry-over works)
    assert np.allclose(prior.page_vector("about"), vectors[0])
    assert np.allclose(prior.chunk_vector("ranking_top_n--best", thesis_hash), vectors[1])

    # carried chunk markdown reconstructs into a wiki-page dict
    assert "ranking_top_n--best" in prior.chunk_md
    parsed = inc.parse_chunk_md(prior.chunk_md["ranking_top_n--best"])
    assert parsed["frontmatter"]["archetype_id"] == "ranking_top_n--best"
    assert parsed["frontmatter"]["evidence_slugs"] == ["about"]


def test_load_prior_bundle_returns_none_without_state():
    # A pre-feature bundle (no state.json) => None => caller falls back to full.
    pages = [_page("about", "About", "x")]
    artifact = export_compiled_wiki(
        pages=pages, chunk_pages=[], thesis_md="# T", thesis_hash="T",
        embedding_vectors=np.zeros((1, 4), dtype=np.float32), embedding_slug_order=["about"],
        manifest={}, state_json=None,
    )
    assert inc.load_prior_bundle(artifact.zip_bytes) is None
