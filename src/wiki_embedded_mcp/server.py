"""MCP server entry point for Wiki Embedded.

Exposes a persistent wiki to MCP clients (Claude Desktop, Cursor, Cline) via:
- 9 Tools         (retrieval + read + graph + provenance + thesis)
- N Resources     (every wiki page as ``wiki-embedded://<slug>``)
- 3 Prompts       (summarize_wiki, ask_about, compare_topics)

Two load modes:
- ``--wiki <dir>``           fresh-parses a local markdown directory and embeds on boot
- ``--compiled-wiki <src>``  loads a precompiled bundle (URL or path) — instant boot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    Resource,
    TextContent,
    Tool,
)

from . import __version__
from ._logging import get_logger
from .compile import compile_wiki
from .embedder import EmbedderConfigError
from .index import DEFAULT_EMBEDDER, WikiIndex
from .loader import CompiledWikiError, LoadedWiki, load_compiled_wiki
from .prompts import PROMPTS, render_prompt
from .resources import (
    SPECIAL_MANIFEST,
    SPECIAL_THESIS,
    build_resources,
    read_resource_payload,
    slug_to_uri,
)

log = get_logger("server")


# ── State ────────────────────────────────────────────────────────────────────
class State:
    """Server-wide state — wiki, index, thesis, manifest."""

    def __init__(self, wiki_dir: Path | None = None, compiled_source: str | None = None):
        if compiled_source is not None:
            self._init_from_compiled(compiled_source)
        elif wiki_dir is not None:
            self._init_from_dir(wiki_dir)
        else:
            raise ValueError("Either wiki_dir or compiled_source must be provided")
        self.pages_by_slug: dict[str, dict[str, Any]] = {
            p["slug"]: p for p in self.compiled["pages"]
        }

    def _init_from_dir(self, wiki_dir: Path) -> None:
        self.mode = "fresh"
        self.wiki_dir = wiki_dir
        self.compiled = compile_wiki(wiki_dir)
        self.index = WikiIndex(self.compiled, embedder_name=DEFAULT_EMBEDDER)
        self.thesis_path: Path | None = wiki_dir / "thesis.md"
        self._thesis_cached: dict | None = None
        self.embedding_model = DEFAULT_EMBEDDER
        self.manifest: dict[str, Any] = {}
        log.info("loaded fresh wiki: %s (%d pages)", wiki_dir, len(self.compiled["pages"]))

    def _init_from_compiled(self, compiled_source: str) -> None:
        self.mode = "precomputed"
        loaded: LoadedWiki = load_compiled_wiki(compiled_source)
        self.wiki_dir = loaded.extract_dir
        self.compiled = loaded.compiled
        embedder_name = loaded.embedding_model or DEFAULT_EMBEDDER
        self.index = WikiIndex(self.compiled, embedder_name=embedder_name)
        self.index.set_precomputed(loaded.embeddings, loaded.slug_order)
        self.thesis_path = loaded.extract_dir / "thesis.md"
        self._thesis_cached = {
            "text": loaded.thesis_text,
            "metadata": loaded.thesis_meta,
            "exists": bool(loaded.thesis_text or loaded.thesis_meta),
        }
        self.embedding_model = embedder_name
        self.manifest = loaded.manifest

    def get_thesis(self) -> dict[str, Any]:
        """Always return the same shape: {exists, text, metadata}."""
        if self._thesis_cached is not None:
            return self._thesis_cached
        if not self.thesis_path or not self.thesis_path.exists():
            return {"exists": False, "text": "", "metadata": {}}
        import frontmatter
        post = frontmatter.load(self.thesis_path)
        cached = {
            "exists": True,
            "text": post.content,
            "metadata": dict(post.metadata),
        }
        self._thesis_cached = cached
        return cached


STATE: State | None = None
server: Server = Server("wiki-embedded-mcp")


# ── Tool definitions ─────────────────────────────────────────────────────────
TOOLS: list[Tool] = [
    Tool(
        name="query_wiki",
        description=(
            "Retrieve top-K relevant items from the wiki. Returns a mixed list of "
            "answer chunks (pre-computed, answer-ready) and atomic pages (raw source). "
            "Each item is tagged with type: 'chunk' or 'page'. Use this as the default retrieval tool."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="query_wiki_answer",
        description=(
            "Fast answer-ready retrieval (pre-computed chunks only). Returns the single "
            "best chunk with its inline citations. Prefer this when the agent wants a "
            "ready answer instead of synthesizing from evidence."
        ),
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    Tool(
        name="query_wiki_pages",
        description=(
            "Atomic pages only (no chunks). Use when the agent wants raw source content "
            "for deep dive, evidence audit, or to synthesize a custom answer."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="read_wiki_page",
        description="Return the full markdown content of a wiki page (or chunk) given its slug.",
        inputSchema={
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    ),
    Tool(
        name="list_wiki_pages",
        description=(
            "Enumerate wiki pages, optionally filtered by frontmatter `kind` "
            "(e.g. 'chunk', 'page', 'segment'). Returns slug + title + kind triples."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Optional frontmatter kind filter."},
                "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000},
            },
        },
    ),
    Tool(
        name="get_thesis",
        description=(
            "Return the wiki's thesis (purpose lens: target audience, primary use case, "
            "answer style, key concepts, excluded topics). Call once per session to calibrate tone."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_provenance",
        description=(
            "Return provenance trail for a wiki page or chunk: kind, cited sources, "
            "LLM teacher (if a chunk), quality_score, thesis_hash, generated_sha."
        ),
        inputSchema={
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    ),
    Tool(
        name="get_wiki_backlinks",
        description=(
            "Return the slugs that cite this slug (reverse crossref graph). "
            "Useful for navigating a wiki by 'who references this concept'."
        ),
        inputSchema={
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
    ),
    Tool(
        name="get_wiki_graph",
        description=(
            "BFS the wiki crossref graph from a starting slug. Returns slugs grouped by "
            "depth level (1 hop, 2 hops, ...). Useful for 'topic neighborhood' exploration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "depth": {"type": "integer", "default": 1, "minimum": 1, "maximum": 5},
            },
            "required": ["slug"],
        },
    ),
]


# ── Handlers ─────────────────────────────────────────────────────────────────
@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if STATE is None:
        return _err("server not initialized")
    log.debug("call_tool name=%s args=%s", name, arguments)
    try:
        if name == "query_wiki":
            return _query_wiki(arguments.get("query", ""), int(arguments.get("top_k", 5)), kind=None)
        if name == "query_wiki_answer":
            return _query_wiki(arguments.get("query", ""), top_k=1, kind="chunk")
        if name == "query_wiki_pages":
            return _query_wiki(arguments.get("query", ""), int(arguments.get("top_k", 5)), kind="page")
        if name == "read_wiki_page":
            return _read_page(arguments.get("slug", ""))
        if name == "list_wiki_pages":
            return _list_pages(arguments.get("kind"), int(arguments.get("limit", 100)))
        if name == "get_thesis":
            return _get_thesis()
        if name == "get_provenance":
            return _get_provenance(arguments.get("slug", ""))
        if name == "get_wiki_backlinks":
            return _get_backlinks(arguments.get("slug", ""))
        if name == "get_wiki_graph":
            return _get_graph(arguments.get("slug", ""), int(arguments.get("depth", 1)))
        return _err(f"unknown tool: {name}")
    except EmbedderConfigError as e:
        log.error("embedder error in %s: %s", name, e)
        return _err(f"embedder not configured: {e}")
    except Exception as e:  # never crash the server on a tool error
        log.exception("tool %s failed", name)
        return _err(f"{type(e).__name__}: {e}")


# ── Resources ────────────────────────────────────────────────────────────────
@server.list_resources()
async def list_resources() -> list[Resource]:
    if STATE is None:
        return []
    thesis_exists = STATE.get_thesis().get("exists", False)
    return build_resources(STATE.compiled["pages"], thesis_exists=thesis_exists)


@server.read_resource()
async def read_resource(uri: str) -> str:
    if STATE is None:
        raise RuntimeError("server not initialized")
    log.debug("read_resource uri=%s", uri)
    return read_resource_payload(
        uri,
        pages_by_slug=STATE.pages_by_slug,
        thesis=STATE.get_thesis() if uri == SPECIAL_THESIS else None,
        manifest=STATE.manifest if uri == SPECIAL_MANIFEST else None,
    )


# ── Prompts ──────────────────────────────────────────────────────────────────
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return PROMPTS


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, Any] | None) -> GetPromptResult:
    log.debug("get_prompt name=%s args=%s", name, arguments)
    return render_prompt(name, arguments)


# ── Tool implementations ─────────────────────────────────────────────────────
def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}, ensure_ascii=False))]


def _ok(payload: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, default=str))]


def _query_wiki(query: str, top_k: int, kind: str | None) -> list[TextContent]:
    if not query.strip():
        return _err("query required")
    assert STATE is not None
    # Over-fetch to keep recall after kind filtering
    candidates = STATE.index.search(query, top_k=top_k * 3 if kind else top_k)
    results: list[dict[str, Any]] = []
    for slug, score in candidates:
        page = STATE.index.get_page(slug)
        if not page:
            continue
        fm = page.get("frontmatter") or {}
        page_kind = fm.get("kind", "page")
        item_type = "chunk" if page_kind == "chunk" else "page"
        if kind == "chunk" and item_type != "chunk":
            continue
        if kind == "page" and item_type != "page":
            continue
        item: dict[str, Any] = {
            "type": item_type,
            "slug": slug,
            "uri": slug_to_uri(slug),
            "title": page.get("title", ""),
            "score": round(score, 4),
        }
        if item_type == "chunk":
            item["answer"] = page.get("body", "")
            item["cites"] = list(fm.get("cites") or [])
        else:
            item["snippet"] = (page.get("body") or "")[:400]
        results.append(item)
        if len(results) >= top_k:
            break
    return _ok({"results": results, "query": query, "kind": kind})


def _read_page(slug: str) -> list[TextContent]:
    assert STATE is not None
    page = STATE.index.get_page(slug)
    if not page:
        return _err(f"slug not found: {slug}")
    return _ok({
        "slug": slug,
        "uri": slug_to_uri(slug),
        "title": page.get("title", ""),
        "body": page.get("body", ""),
        "frontmatter": page.get("frontmatter", {}),
    })


def _list_pages(kind: str | None, limit: int) -> list[TextContent]:
    assert STATE is not None
    items: list[dict[str, Any]] = []
    for p in STATE.compiled["pages"]:
        fm = p.get("frontmatter") or {}
        p_kind = fm.get("kind", "page")
        if kind and p_kind != kind:
            continue
        items.append({"slug": p["slug"], "title": p.get("title", ""), "kind": p_kind})
        if len(items) >= limit:
            break
    return _ok({"pages": items, "total": len(items), "kind_filter": kind})


def _get_thesis() -> list[TextContent]:
    assert STATE is not None
    return _ok(STATE.get_thesis())


def _get_provenance(slug: str) -> list[TextContent]:
    assert STATE is not None
    page = STATE.index.get_page(slug)
    if not page:
        return _err(f"slug not found: {slug}")
    fm = page.get("frontmatter") or {}
    return _ok({
        "slug": slug,
        "kind": fm.get("kind", "page"),
        "title": page.get("title", ""),
        "cites": list(fm.get("cites") or []),
        "evidence_slugs": list(fm.get("evidence_slugs") or []),
        "teacher": fm.get("teacher"),
        "quality_score": fm.get("quality_score"),
        "generated_sha": fm.get("generated_sha"),
        "last_verified_sha": fm.get("last_verified_sha"),
        "updated_at": fm.get("updated_at"),
        "thesis_hash": fm.get("thesis_hash"),
    })


def _get_backlinks(slug: str) -> list[TextContent]:
    if not slug:
        return _err("slug required")
    assert STATE is not None
    if slug not in STATE.compiled["slug_set"]:
        return _err(f"slug not found: {slug}")
    backs = STATE.index.get_backlinks(slug)
    return _ok({"slug": slug, "backlinks": backs, "count": len(backs)})


def _get_graph(slug: str, depth: int) -> list[TextContent]:
    if not slug:
        return _err("slug required")
    assert STATE is not None
    if slug not in STATE.compiled["slug_set"]:
        return _err(f"slug not found: {slug}")
    depth = max(1, min(depth, 5))
    neighborhood = STATE.index.get_neighborhood(slug, depth=depth)
    forward = STATE.index.get_forward_links(slug)
    backlinks = STATE.index.get_backlinks(slug)
    return _ok({
        "slug": slug,
        "depth": depth,
        "forward_links": forward,
        "backlinks": backlinks,
        "neighborhood_by_depth": neighborhood,
    })


# ── Entrypoint ───────────────────────────────────────────────────────────────
async def run(wiki_dir: Path | None, compiled_source: str | None) -> None:
    global STATE
    STATE = State(wiki_dir=wiki_dir, compiled_source=compiled_source)
    # Fresh-mode pre-embeds at startup; precomputed-mode is already loaded.
    if STATE.mode == "fresh":
        STATE.index.embed_pages()
    log.info(
        "wiki-embedded-mcp v%s ready (mode=%s, pages=%d, embedder=%s)",
        __version__, STATE.mode, len(STATE.compiled["pages"]), STATE.embedding_model,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(
                notification_options=NotificationOptions(),
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wiki-embedded-mcp",
        description=(
            "MCP server for persistent knowledge wikis. "
            "Loads from a local directory OR a compiled_wiki.zip produced by 42rows-wiki-compiler."
        ),
    )
    parser.add_argument("-V", "--version", action="version", version=f"wiki-embedded-mcp {__version__}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--wiki",
        type=Path,
        default=None,
        help="Path to a wiki directory (markdown files). Parsed and embedded at startup.",
    )
    group.add_argument(
        "--compiled-wiki",
        dest="compiled_wiki",
        type=str,
        default=None,
        help="URL or path to a compiled_wiki.zip (instant boot, no re-embedding).",
    )
    args = parser.parse_args()

    # Resolve from env if neither flag given
    wiki_dir = args.wiki
    compiled_source = args.compiled_wiki
    if wiki_dir is None and compiled_source is None:
        env_compiled = os.environ.get("WIKI_EMBEDDED_COMPILED")
        env_dir = os.environ.get("WIKI_EMBEDDED_DIR")
        if env_compiled:
            compiled_source = env_compiled
        elif env_dir:
            wiki_dir = Path(env_dir)
        else:
            wiki_dir = Path("./wiki")

    if compiled_source is None and wiki_dir is not None and not wiki_dir.exists():
        log.error("wiki dir not found: %s", wiki_dir)
        raise SystemExit(1)

    try:
        asyncio.run(run(wiki_dir, compiled_source))
    except (CompiledWikiError, EmbedderConfigError) as e:
        log.error("startup failed: %s", e)
        raise SystemExit(2) from e


if __name__ == "__main__":
    main()
