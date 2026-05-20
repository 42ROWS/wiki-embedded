"""Tests for compile.compile_wiki — markdown parsing + crossref extraction."""
from __future__ import annotations

from pathlib import Path

from wiki_embedded_mcp.compile import compile_wiki, extract_crossrefs


def test_compile_parses_all_pages(sample_wiki: Path) -> None:
    compiled = compile_wiki(sample_wiki)
    slugs = {p["slug"] for p in compiled["pages"]}
    # 4 pages + 1 chunk (thesis.md is a wiki page too with slug="thesis")
    assert {"pages/alpha", "pages/beta", "pages/gamma", "pages/delta",
            "chunks/abc123/ranking-top"} <= slugs


def test_compile_extracts_wikilinks(sample_wiki: Path) -> None:
    compiled = compile_wiki(sample_wiki)
    by_slug = {p["slug"]: p for p in compiled["pages"]}
    # alpha → beta (single forward link)
    assert "pages/beta" in by_slug["pages/alpha"]["crossrefs"]
    # gamma → alpha and gamma → delta
    assert "pages/alpha" in by_slug["pages/gamma"]["crossrefs"]
    assert "pages/delta" in by_slug["pages/gamma"]["crossrefs"]


def test_compile_filters_graph_to_in_corpus(sample_wiki: Path) -> None:
    compiled = compile_wiki(sample_wiki)
    # No edge should point to a slug not in slug_set
    for src, targets in compiled["graph"].items():
        for tgt in targets:
            assert tgt in compiled["slug_set"], f"dangling edge {src} → {tgt}"


def test_extract_crossrefs_handles_pipe_form() -> None:
    body = "See [[pages/alpha|the alpha page]] and [[pages/beta]]."
    refs = extract_crossrefs(body, {})
    assert refs == {"pages/alpha", "pages/beta"}


def test_extract_crossrefs_reads_frontmatter_fields() -> None:
    fm = {
        "cites": ["pages/alpha", "pages/beta"],
        "sources": [{"slug": "pages/gamma"}],
    }
    refs = extract_crossrefs("", fm)
    assert refs == {"pages/alpha", "pages/beta", "pages/gamma"}
