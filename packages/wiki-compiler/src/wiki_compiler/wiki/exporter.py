"""Export a compiled wiki into a portable .zip artifact + MCP manifest.

The artifact layout (compiled_wiki.zip):
    pages/
        <slug>.md            # original frontmatter + body, slug-as-path
    chunks/
        <thesis_hash>/
            <archetype_id>.md  # pre-computed answer chunks
    thesis.md                # derived thesis with frontmatter
    embeddings.npz           # numpy archive: vectors + slug_order
    manifest.json            # everything needed to bootstrap the MCP server

mario-wiki-mcp v0.2+ reads this layout directly via `--compiled-wiki <path|url>`.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from typing import Any

import frontmatter
import numpy as np

SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_./-]+")


def _safe_path(slug: str) -> str:
    """Normalize a slug into a relative POSIX-style path safe for zip entries."""
    s = SAFE_FILENAME_RE.sub("-", slug.strip("/"))
    return s.strip("-/") or "page"


def _page_to_md(page: dict[str, Any]) -> str:
    fm = dict(page.get("frontmatter") or {})
    fm.setdefault("slug", page["slug"])
    if page.get("title"):
        fm.setdefault("title", page["title"])
    post = frontmatter.Post(page["body"], **fm)
    return frontmatter.dumps(post)


@dataclass
class ExportedArtifact:
    zip_bytes: bytes
    manifest: dict[str, Any]


def build_manifest(
    *,
    pages_count: int,
    chunks_count: int,
    crossref_edges: int,
    thesis_hash: str,
    thesis_summary: str,
    embedding_model: str,
    embedding_dims: int,
    source_kind: str,
    source_origin: str,
    answer_style: str,
    chunk_archetypes: list[str],
    teacher_model: str,
    mcp_server_pip: str = "mariowiki-mcp>=0.2",
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "compiler": "42rows-wiki-compiler",
        "compiler_version": "0.1",
        "source": {"kind": source_kind, "origin": source_origin},
        "wiki": {
            "pages_count": pages_count,
            "chunks_count": chunks_count,
            "crossref_edges": crossref_edges,
        },
        "thesis": {"hash": thesis_hash, "summary": thesis_summary},
        "embeddings": {"model": embedding_model, "dims": embedding_dims},
        "chunks_config": {
            "answer_style": answer_style,
            "archetypes": chunk_archetypes,
            "teacher_model": teacher_model,
        },
        "mcp": {
            "consumer_pip": mcp_server_pip,
            "config_snippet": {
                "mcpServers": {
                    "mariowiki": {
                        "command": "mariowiki-mcp",
                        "args": ["--compiled-wiki", "<PATH_OR_URL>"],
                    }
                }
            },
        },
    }


def export_compiled_wiki(
    *,
    pages: list[dict[str, Any]],
    chunk_pages: list[dict[str, Any]],
    thesis_md: str,
    thesis_hash: str,
    embedding_vectors: np.ndarray,
    embedding_slug_order: list[str],
    manifest: dict[str, Any],
    state_json: str | None = None,
) -> ExportedArtifact:
    """Produce the compiled_wiki.zip in-memory along with the manifest dict.

    ``state_json`` (the diff baseline) is written as ``state.json`` when provided;
    it is what enables incremental updates on the next compile.
    """
    buf = io.BytesIO()
    seen_paths: set[str] = set()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # pages/<slug>.md
        for p in pages:
            rel = f"pages/{_safe_path(p['slug'])}.md"
            rel = _dedup_path(rel, seen_paths)
            zf.writestr(rel, _page_to_md(p))

        # chunks/<thesis_hash>/<slug>.md
        for c in chunk_pages:
            slug = c["slug"]
            base = _safe_path(slug).replace(f"chunks/{thesis_hash}/", "")
            rel = f"chunks/{thesis_hash}/{base}.md"
            rel = _dedup_path(rel, seen_paths)
            zf.writestr(rel, _page_to_md(c))

        # thesis.md
        zf.writestr("thesis.md", thesis_md)

        # embeddings.npz
        npz_buf = io.BytesIO()
        np.savez_compressed(
            npz_buf,
            vectors=embedding_vectors.astype(np.float32),
            slug_order=np.array(embedding_slug_order, dtype=object),
        )
        zf.writestr("embeddings.npz", npz_buf.getvalue())

        # manifest.json
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

        # state.json — diff baseline for incremental updates (optional)
        if state_json is not None:
            zf.writestr("state.json", state_json)

    return ExportedArtifact(zip_bytes=buf.getvalue(), manifest=manifest)


def _dedup_path(path: str, seen: set[str]) -> str:
    """Ensure unique zip member paths even on slug collisions."""
    if path not in seen:
        seen.add(path)
        return path
    base, _, ext = path.rpartition(".")
    i = 2
    while True:
        cand = f"{base}-{i}.{ext}" if ext else f"{path}-{i}"
        if cand not in seen:
            seen.add(cand)
            return cand
        i += 1
