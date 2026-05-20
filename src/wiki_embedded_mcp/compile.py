"""Wiki compilation: parse markdown pages with frontmatter + extract crossrefs.

Parse markdown wiki + extract crossrefs (frontmatter + [[wikilinks]]).
"""
from __future__ import annotations

import re
from pathlib import Path

import frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
SKIP_DIRS = {"_meta", ".plugs"}
SKIP_FILES = {"log.md"}


def extract_crossrefs(body: str, fm: dict) -> set[str]:
    """Extract slug references from frontmatter + body."""
    refs: set[str] = set()
    for field in ("cross_references", "sources", "evidence", "cites"):
        v = fm.get(field) or []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    t = item.get("target") or item.get("slug")
                    if t:
                        refs.add(str(t).strip())
                elif isinstance(item, str):
                    refs.add(item.split("#")[0].strip())
    for m in WIKILINK_RE.finditer(body):
        s = m.group(1).strip().lstrip("/")
        if s:
            refs.add(s)
    return {r for r in refs if r}


def compile_wiki(wiki_dir: Path) -> dict:
    """Compile wiki directory into structured dict.

    Returns:
        {
            "pages": [{slug, title, body, content_full, frontmatter, crossrefs}],
            "slug_set": set[str],
            "graph": {slug: set[slug]} adjacency,
            "wiki_dir": str,
        }
    """
    wiki_dir = Path(wiki_dir)
    pages = []
    graph: dict[str, set[str]] = {}
    slug_set: set[str] = set()

    for md in sorted(wiki_dir.rglob("*.md")):
        rel = md.relative_to(wiki_dir)
        if any(p in SKIP_DIRS for p in rel.parts) or md.name in SKIP_FILES:
            continue
        try:
            post = frontmatter.load(md)
        except Exception:
            continue

        slug = str(post.get("slug") or str(rel).replace(".md", ""))
        title = str(post.get("title") or "")
        body = post.content.strip()
        crossrefs = extract_crossrefs(body, post.metadata)

        pages.append({
            "slug": slug,
            "title": title,
            "body": body,
            "content_full": f"{title}\n\n{body}",
            "frontmatter": dict(post.metadata),
            "crossrefs": crossrefs,
        })
        graph[slug] = crossrefs
        slug_set.add(slug)

    # Filter graph: keep only edges to existing slugs
    for s in graph:
        graph[s] = {r for r in graph[s] if r in slug_set and r != s}

    return {
        "pages": pages,
        "graph": graph,
        "slug_set": slug_set,
        "wiki_dir": str(wiki_dir),
    }
