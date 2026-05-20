"""Tests for index.WikiIndex — graph helpers + precomputed mode."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wiki_embedded_mcp.compile import compile_wiki
from wiki_embedded_mcp.index import WikiIndex


def _make_index(sample_wiki: Path) -> WikiIndex:
    compiled = compile_wiki(sample_wiki)
    return WikiIndex(compiled, embedder_name="test:dummy")


def test_get_forward_links(sample_wiki: Path) -> None:
    idx = _make_index(sample_wiki)
    assert idx.get_forward_links("pages/alpha") == ["pages/beta"]
    assert idx.get_forward_links("pages/gamma") == ["pages/alpha", "pages/delta"]


def test_get_backlinks(sample_wiki: Path) -> None:
    idx = _make_index(sample_wiki)
    # alpha is cited by beta, gamma, and the chunk
    assert set(idx.get_backlinks("pages/alpha")) >= {"pages/beta", "pages/gamma"}
    # delta is only cited by gamma
    assert idx.get_backlinks("pages/delta") == ["pages/gamma"]


def test_get_neighborhood_depth1(sample_wiki: Path) -> None:
    idx = _make_index(sample_wiki)
    nb = idx.get_neighborhood("pages/alpha", depth=1)
    # depth=1 neighbors of alpha = forward (beta) + backlinks (beta, gamma, chunk)
    assert "1" in nb
    assert "pages/beta" in nb["1"]
    assert "pages/gamma" in nb["1"]


def test_set_precomputed_validates_shape(sample_wiki: Path) -> None:
    idx = _make_index(sample_wiki)
    bad_vectors = np.zeros((3, 4), dtype=np.float32)
    bad_slugs = ["a", "b"]  # length mismatch
    try:
        idx.set_precomputed(bad_vectors, bad_slugs)
    except ValueError as e:
        assert "vectors rows" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_search_uses_precomputed(sample_wiki: Path) -> None:
    """Search with precomputed embeddings should not invoke the embedder.

    We inject fake vectors so cosine math is deterministic, then monkey-patch
    embed_query to return one of them.
    """
    idx = _make_index(sample_wiki)
    pages = [p["slug"] for p in idx.pages]
    # 4-D one-hot vectors so cosine ranking is exact
    dim = 4
    vectors = np.eye(min(len(pages), dim), dim, dtype=np.float32)
    while vectors.shape[0] < len(pages):
        vectors = np.vstack([vectors, np.zeros((1, dim), dtype=np.float32)])
    vectors = vectors[: len(pages)]
    idx.set_precomputed(vectors, pages)

    # Monkey-patch embed_query to return vector matching the FIRST page
    idx.embed_query = lambda q: vectors[0]

    results = idx.search("anything", top_k=1)
    assert results
    top_slug, top_score = results[0]
    assert top_slug == pages[0]
    assert top_score > 0.99
