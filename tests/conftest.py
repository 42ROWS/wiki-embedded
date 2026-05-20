"""Shared pytest fixtures for wiki-embedded-mcp tests.

The fixtures build a tiny in-memory wiki on disk (5 pages + 1 chunk + thesis)
and a precomputed-bundle zip so we can exercise both load modes without
hitting an embedder or the network.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest


# ── helper writers ────────────────────────────────────────────────────────────
def _md(slug: str, title: str, body: str, **fm) -> str:
    """Write a markdown file with frontmatter. JSON-encode strings to keep YAML happy
    even when values contain colons or other YAML metacharacters."""
    fm.setdefault("slug", slug)
    fm.setdefault("title", title)
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, (list, dict)):
            lines.append(f"{k}: {json.dumps(v)}")
        elif isinstance(v, str):
            # Quote strings to dodge YAML special chars (colons, hashes, etc.)
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


# ── sample wiki dir ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_wiki(tmp_path: Path) -> Path:
    """Build a tiny wiki on disk: 5 pages with crossrefs + a thesis."""
    pages = tmp_path / "pages"
    pages.mkdir()

    (pages / "alpha.md").write_text(_md(
        "pages/alpha",
        "Alpha — core concept",
        dedent("""
            Alpha is the foundational concept that everything builds on.
            See [[pages/beta]] for the practical follow-up.
        """).strip(),
    ))
    (pages / "beta.md").write_text(_md(
        "pages/beta",
        "Beta — practical follow-up",
        "Beta extends Alpha with the concrete pipeline. References [[pages/alpha]].",
    ))
    (pages / "gamma.md").write_text(_md(
        "pages/gamma",
        "Gamma — orthogonal topic",
        "Gamma stands on its own. Cross-cites [[pages/alpha]] and [[pages/delta]].",
    ))
    (pages / "delta.md").write_text(_md(
        "pages/delta",
        "Delta — endpoint",
        "Delta is the terminal node. No outgoing refs.",
    ))

    chunks = tmp_path / "chunks" / "abc123"
    chunks.mkdir(parents=True)
    (chunks / "ranking-top.md").write_text(_md(
        "chunks/abc123/ranking-top",
        "Top concepts to learn first",
        dedent("""
            To start, read [[pages/alpha]] then [[pages/beta]]. Skip
            [[pages/gamma]] unless you need orthogonal coverage.
        """).strip(),
        kind="chunk",
        cites=["pages/alpha", "pages/beta"],
        evidence_slugs=["pages/alpha", "pages/beta", "pages/gamma"],
        teacher="test-model",
        quality_score=88.0,
        thesis_hash="abc123",
    ))

    (tmp_path / "thesis.md").write_text(_md(
        "thesis",
        "Thesis: getting started",
        "Help new users move from Alpha to Beta with the right scaffolding.",
        kind="thesis",
        target_audience="new users",
        primary_use_case="onboarding from zero",
        answer_style="concise, tactical",
        key_concepts=["alpha", "beta", "scaffolding"],
        excluded_topics=["billing"],
        thesis_hash="abc123",
    ))
    return tmp_path


# ── sample compiled_wiki.zip ──────────────────────────────────────────────────
@pytest.fixture
def sample_compiled_bundle(sample_wiki: Path, tmp_path: Path) -> Path:
    """Build a compiled_wiki.zip with embeddings + manifest using the sample wiki."""
    bundle_path = tmp_path / "compiled_wiki.zip"

    # Walk all .md in sample_wiki preserving relative paths
    files: list[tuple[str, bytes]] = []
    for md in sorted(sample_wiki.rglob("*.md")):
        rel = md.relative_to(sample_wiki)
        files.append((str(rel).replace("\\", "/"), md.read_bytes()))

    # Synthetic precomputed embeddings: 5 vectors × 4 dims, L2-normalized
    slugs = [
        "pages/alpha",
        "pages/beta",
        "pages/gamma",
        "pages/delta",
        "chunks/abc123/ranking-top",
    ]
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(len(slugs), 4)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = (raw / norms).astype(np.float32)

    npz_buf = io.BytesIO()
    np.savez_compressed(
        npz_buf,
        vectors=vectors,
        slug_order=np.array(slugs, dtype=object),
    )

    manifest = {
        "format_version": 1,
        "compiler": "test-fixture",
        "compiler_version": "0.0.0",
        "source": {"kind": "fixture", "origin": "pytest"},
        "wiki": {"pages_count": 4, "chunks_count": 1, "crossref_edges": 4},
        "thesis": {"hash": "abc123", "summary": "test thesis"},
        "embeddings": {"model": "test:tiny-4d", "dims": 4},
    }

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files:
            zf.writestr(name, content)
        zf.writestr("embeddings.npz", npz_buf.getvalue())
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return bundle_path
