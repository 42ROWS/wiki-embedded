"""MCP Resources surface — expose wiki pages as readable URIs.

Resources let clients browse and read wiki content directly via the URI scheme
``wiki-embedded://<slug>``, without going through tool calls. Pattern used by
canonical Anthropic MCP servers (filesystem, github, postgres).

Special URIs:
    wiki-embedded://thesis       — the wiki's purpose lens (frontmatter + body)
    wiki-embedded://__manifest__ — compiled-wiki bundle manifest (precomputed mode)
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote, urlparse

from mcp.types import Resource

URI_SCHEME = "wiki-embedded"
SPECIAL_THESIS = f"{URI_SCHEME}://thesis"
SPECIAL_MANIFEST = f"{URI_SCHEME}://__manifest__"


def slug_to_uri(slug: str) -> str:
    """Convert an internal wiki slug into a stable MCP resource URI."""
    safe = slug.strip("/").replace(" ", "%20")
    return f"{URI_SCHEME}://{safe}"


def uri_to_slug(uri: str) -> str | None:
    """Parse a wiki-embedded:// URI back into a wiki slug. Returns None on mismatch."""
    parsed = urlparse(uri)
    if parsed.scheme != URI_SCHEME:
        return None
    # urlparse on custom scheme puts everything in `netloc` + `path`
    raw = (parsed.netloc + parsed.path).strip("/")
    return unquote(raw) if raw else None


def build_resources(pages: list[dict[str, Any]], thesis_exists: bool) -> list[Resource]:
    """Build the static list of resources exposed by this server."""
    resources: list[Resource] = []
    if thesis_exists:
        resources.append(
            Resource(
                uri=SPECIAL_THESIS,
                name="Wiki Thesis",
                description="Purpose lens of the wiki (target audience, use case, style).",
                mimeType="text/markdown",
            )
        )
    resources.append(
        Resource(
            uri=SPECIAL_MANIFEST,
            name="Compiled Bundle Manifest",
            description="JSON manifest of the loaded compiled_wiki bundle (if any).",
            mimeType="application/json",
        )
    )
    for p in pages:
        fm = p.get("frontmatter") or {}
        kind = fm.get("kind", "page")
        title = (p.get("title") or p["slug"])[:120]
        resources.append(
            Resource(
                uri=slug_to_uri(p["slug"]),
                name=f"[{kind}] {title}",
                description=f"Wiki {kind}: {p['slug']}",
                mimeType="text/markdown",
            )
        )
    return resources


def read_resource_payload(
    uri: str,
    pages_by_slug: dict[str, dict[str, Any]],
    thesis: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> str:
    """Return the textual content for a given wiki-embedded:// URI."""
    if uri == SPECIAL_THESIS:
        if not thesis or not thesis.get("exists"):
            return "No thesis defined for this wiki."
        meta = thesis.get("metadata") or {}
        body = thesis.get("text") or ""
        return json.dumps({"metadata": meta, "text": body}, ensure_ascii=False, indent=2, default=str)
    if uri == SPECIAL_MANIFEST:
        return json.dumps(manifest or {}, ensure_ascii=False, indent=2, default=str)

    slug = uri_to_slug(uri)
    if slug is None:
        raise ValueError(f"Not a wiki-embedded:// URI: {uri!r}")
    page = pages_by_slug.get(slug)
    if page is None:
        raise FileNotFoundError(f"slug not found: {slug}")
    # Markdown source: serialize body + frontmatter as a single markdown document
    fm = page.get("frontmatter") or {}
    fm_yaml = "\n".join(f"{k}: {json.dumps(v, default=str, ensure_ascii=False)}" for k, v in fm.items())
    fm_block = f"---\n{fm_yaml}\n---\n\n" if fm_yaml else ""
    title_block = f"# {page['title']}\n\n" if page.get("title") else ""
    return f"{fm_block}{title_block}{page.get('body', '')}"
