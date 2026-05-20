"""Parse a wiki directory of markdown files into structured pages + crossref graph.

Ported pattern from mario-wiki-mcp/src/mariowiki_mcp/compile.py.
Pages are dicts with: slug, title, body, content_full, frontmatter (dict), crossrefs (set).
Graph is adjacency: {slug: set[slug]} restricted to in-corpus targets.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
SKIP_DIRS = {"_meta", ".plugs", ".git", ".github", "node_modules", "__pycache__"}
SKIP_FILES = {"log.md"}


@dataclass
class CompiledWiki:
    pages: list[dict[str, Any]] = field(default_factory=list)
    graph: dict[str, set[str]] = field(default_factory=dict)
    slug_set: set[str] = field(default_factory=set)
    wiki_dir: str = ""
    parse_errors: list[str] = field(default_factory=list)

    @property
    def pages_count(self) -> int:
        return len(self.pages)

    @property
    def crossref_edges(self) -> int:
        return sum(len(v) for v in self.graph.values())


def extract_crossrefs(body: str, fm: dict[str, Any]) -> set[str]:
    """Extract slug references from frontmatter fields + body wikilinks."""
    refs: set[str] = set()
    for field_name in ("cross_references", "sources", "evidence", "cites"):
        v = fm.get(field_name) or []
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


def compile_wiki(wiki_dir: Path | str) -> CompiledWiki:
    """Scan a directory for .md files and compile them into a structured wiki.

    - Skips SKIP_DIRS / SKIP_FILES
    - Parses YAML frontmatter (graceful on errors)
    - Extracts crossrefs from frontmatter + body [[slug]] wikilinks
    - Filters crossref graph to in-corpus slugs only

    Returns a CompiledWiki dataclass.
    """
    wiki_dir = Path(wiki_dir)
    result = CompiledWiki(wiki_dir=str(wiki_dir))

    for md in sorted(wiki_dir.rglob("*.md")):
        rel = md.relative_to(wiki_dir)
        if any(p in SKIP_DIRS for p in rel.parts) or md.name in SKIP_FILES:
            continue
        try:
            post = frontmatter.load(md)
        except Exception as e:
            result.parse_errors.append(f"{rel}: {e}")
            continue

        slug = str(post.get("slug") or str(rel).replace(".md", ""))
        title = str(post.get("title") or "")
        body = post.content.strip()
        crossrefs = extract_crossrefs(body, post.metadata)

        result.pages.append({
            "slug": slug,
            "title": title,
            "body": body,
            "content_full": f"{title}\n\n{body}",
            "frontmatter": dict(post.metadata),
            "crossrefs": crossrefs,
        })
        result.graph[slug] = crossrefs
        result.slug_set.add(slug)

    # Restrict graph to in-corpus targets (no dangling refs)
    for s in result.graph:
        result.graph[s] = {r for r in result.graph[s] if r in result.slug_set and r != s}

    return result


def sample_pages_for_thesis(wiki: CompiledWiki, n: int = 50) -> list[dict[str, Any]]:
    """Stratified sample of pages for thesis derivation.

    Sampled across `kind` frontmatter values when available; otherwise uniform.
    """
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for p in wiki.pages:
        k = p["frontmatter"].get("kind") or "default"
        by_kind.setdefault(k, []).append(p)

    kinds = list(by_kind.keys())
    if not kinds:
        return []

    per_kind = max(1, n // len(kinds))
    out: list[dict[str, Any]] = []
    for k in kinds:
        out.extend(by_kind[k][:per_kind])
    return out[:n]
