"""Smoke test for v0.2 — load a compiled_wiki.zip and exercise the MCP tools.

Usage:
    PINECONE_API_KEY=... python examples/test_compiled_bundle.py <URL_or_path>

Without arguments, falls back to a published example bundle.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path

# Make the source tree importable when running from a fresh checkout
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


DEFAULT_BUNDLE = (
    "https://api.apify.com/v2/key-value-stores/gLcV2yHHuB6kVU8e2/records/"
    "compiled_wiki.zip?signature=1bIdhM8Z4cfEehuUIjnu8"
)


async def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BUNDLE
    print(f"=== wiki-embedded-mcp v0.2 — compiled-bundle smoke test ===\n")
    print(f"source: {source}\n")

    from wiki_embedded_mcp import server as srv

    print("[1/5] Booting server in precomputed mode...")
    srv.STATE = srv.State(compiled_source=source)
    pages = srv.STATE.compiled["pages"]
    print(f"  pages loaded: {len(pages)}")
    print(f"  embedding model: {srv.STATE.embedding_model}")

    async def call(name: str, args: dict) -> dict:
        content = await srv.call_tool(name, args)
        return json.loads(content[0].text)

    print("\n[2/5] get_thesis")
    thesis = await call("get_thesis", {})
    print(f"  exists: {thesis.get('exists')}")
    if thesis.get("exists"):
        print(f"  preview: {thesis['text'][:140]}...")

    print("\n[3/5] list_wiki_pages (kind=chunk, limit=3)")
    chunks_list = await call("list_wiki_pages", {"kind": "chunk", "limit": 3})
    for p in chunks_list.get("pages", []):
        print(f"  - {p['slug']}: {p['title'][:60]}")

    print("\n[4/5] query_wiki_answer — sample queries")
    for q in [
        "best template for dynamic web scraping",
        "cheerio vs playwright",
        "AI agent template python",
    ]:
        r = await call("query_wiki_answer", {"query": q})
        for item in r.get("results", []):
            print(f"  Q: {q!r}")
            print(f"    [{item.get('type')}] {item.get('slug')} score={item.get('score')}")
            print(f"    title: {item.get('title', '')[:70]}")
            print(f"    cites: {item.get('cites', [])[:4]}")
            print()

    print("[5/5] get_wiki_backlinks + get_wiki_graph on first slug")
    first_slug = pages[0]["slug"]
    backs = await call("get_wiki_backlinks", {"slug": first_slug})
    print(f"  {first_slug}: {len(backs.get('backlinks', []))} backlinks")
    graph = await call("get_wiki_graph", {"slug": first_slug, "depth": 1})
    print(f"  neighborhood@1: {len(graph.get('neighborhood_by_depth', {}).get('1', []))} nodes")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    if not os.getenv("PINECONE_API_KEY"):
        print(
            "Warning: PINECONE_API_KEY not set. The bundle likely uses pinecone: embeddings\n"
            "and query embedding will fail. Set the env var first."
        )
    asyncio.run(main())
