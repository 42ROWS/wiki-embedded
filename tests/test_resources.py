"""Tests for resources.py — MCP Resource URI scheme + payloads."""
from __future__ import annotations

from wiki_embedded_mcp.resources import (
    SPECIAL_MANIFEST,
    SPECIAL_THESIS,
    build_resources,
    read_resource_payload,
    slug_to_uri,
    uri_to_slug,
)


def test_uri_roundtrip() -> None:
    slug = "pages/alpha"
    uri = slug_to_uri(slug)
    assert uri == "wiki-embedded://pages/alpha"
    assert uri_to_slug(uri) == slug


def test_uri_with_spaces() -> None:
    slug = "pages/with space"
    uri = slug_to_uri(slug)
    assert uri_to_slug(uri) == slug


def test_uri_rejects_foreign_scheme() -> None:
    assert uri_to_slug("file:///etc/passwd") is None
    assert uri_to_slug("https://example.com") is None


def test_build_resources_includes_specials() -> None:
    resources = build_resources(
        pages=[{"slug": "pages/alpha", "title": "Alpha", "frontmatter": {"kind": "page"}}],
        thesis_exists=True,
    )
    # Resource.uri is a pydantic AnyUrl; coerce to str for comparison
    uris = {str(r.uri) for r in resources}
    assert SPECIAL_THESIS in uris
    assert SPECIAL_MANIFEST in uris
    assert "wiki-embedded://pages/alpha" in uris


def test_read_resource_thesis() -> None:
    out = read_resource_payload(
        SPECIAL_THESIS,
        pages_by_slug={},
        thesis={"exists": True, "text": "T", "metadata": {"k": "v"}},
        manifest=None,
    )
    assert "T" in out
    assert "metadata" in out


def test_read_resource_manifest() -> None:
    out = read_resource_payload(
        SPECIAL_MANIFEST,
        pages_by_slug={},
        thesis=None,
        manifest={"compiler": "test"},
    )
    assert "compiler" in out


def test_read_resource_page() -> None:
    page = {
        "slug": "pages/alpha",
        "title": "Alpha",
        "body": "Body text",
        "frontmatter": {"kind": "page"},
    }
    out = read_resource_payload(
        "wiki-embedded://pages/alpha",
        pages_by_slug={"pages/alpha": page},
        thesis=None,
        manifest=None,
    )
    assert "Alpha" in out
    assert "Body text" in out
