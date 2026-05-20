"""Unit tests for the per-(collection, tenant) Pinecone store.

Uses an in-memory fake index (no live Pinecone), so it verifies routing,
namespace isolation, metadata, and the delta-sync plan deterministically.
See ADR 2026-05-20-incremental-wiki-embedding-update.
"""
from __future__ import annotations

import types

import numpy as np

from wiki_compiler.tenant_store import (
    TenantStore,
    build_metadata,
    index_name_for,
    sync_delta,
)


class FakeIndex:
    """Minimal stand-in for a Pinecone Index, partitioned by namespace."""

    def __init__(self):
        self.namespaces: dict[str, dict[str, tuple]] = {}
        self.calls: list[str] = []

    def upsert(self, vectors, namespace):
        self.calls.append(f"upsert:{namespace}:{len(vectors)}")
        ns = self.namespaces.setdefault(namespace, {})
        for rid, vec, md in vectors:
            ns[rid] = (vec, md)

    def delete(self, ids=None, namespace=None, delete_all=False):
        self.calls.append(f"delete:{namespace}:{'ALL' if delete_all else (ids or [])}")
        ns = self.namespaces.setdefault(namespace, {})
        if delete_all:
            ns.clear()
        elif ids:
            for i in ids:
                ns.pop(i, None)

    def query(self, vector, top_k, include_metadata, namespace):  # noqa: ARG002 (Pinecone-compatible signature)
        ns = self.namespaces.get(namespace, {})
        matches = [
            types.SimpleNamespace(id=rid, score=1.0, metadata=md)
            for rid, (_vec, md) in list(ns.items())[:top_k]
        ]
        return types.SimpleNamespace(matches=matches)


def _store(tenant: str, index: FakeIndex) -> TenantStore:
    return TenantStore("company-wiki", tenant, index=index)


# --------------------------------------------------------------------------- naming

def test_index_name_for_sanitizes():
    assert index_name_for("company-wiki") == "company-wiki"
    assert index_name_for("Company Wiki!") == "company-wiki"
    assert index_name_for("  Marketing_Skills  ") == "marketing-skills"


# --------------------------------------------------------------------------- metadata

def test_build_metadata_page_vs_chunk():
    page = {"slug": "about", "title": "About", "body": "b", "content_full": "About\n\nb", "frontmatter": {}}
    md = build_metadata(page, "h1")
    assert md["kind"] == "page" and md["text"] == "About\n\nb" and md["content_hash"] == "h1"

    chunk = {
        "slug": "chunks/T/x", "title": "X", "body": "answer", "content_full": "X\n\nanswer",
        "frontmatter": {"kind": "chunk", "cites": ["about"], "category": "explanatory", "quality_score": 90.0},
    }
    md = build_metadata(chunk, "h2")
    assert md["kind"] == "chunk" and md["text"] == "answer" and md["cites"] == ["about"]


# --------------------------------------------------------------------------- routing + isolation

def test_upsert_query_routed_to_namespace():
    idx = FakeIndex()
    store = _store("acme", idx)
    store.upsert([("about", np.array([1.0, 0.0], dtype=np.float32), {"kind": "page", "title": "About"})])
    assert "acme" in idx.namespaces and "about" in idx.namespaces["acme"]
    hits = store.query(np.array([1.0, 0.0], dtype=np.float32), top_k=5)
    assert len(hits) == 1 and hits[0].id == "about" and hits[0].title == "About"


def test_tenant_isolation():
    idx = FakeIndex()
    acme = _store("acme", idx)
    globex = _store("globex", idx)
    acme.upsert([("p", np.array([1.0], dtype=np.float32), {})])
    # globex's handle never sees acme's vector
    assert globex.query(np.array([1.0], dtype=np.float32), top_k=5) == []
    assert len(acme.query(np.array([1.0], dtype=np.float32), top_k=5)) == 1


def test_requires_tenant():
    try:
        TenantStore("company-wiki", "", index=FakeIndex())
    except ValueError:
        return
    raise AssertionError("empty tenant should raise")


# --------------------------------------------------------------------------- sync_delta

def _vec(x):
    return np.array([x], dtype=np.float32)


def test_sync_delta_full_resets_then_upserts_all():
    idx = FakeIndex()
    # pre-existing stale vector in the namespace
    idx.namespaces["acme"] = {"old": (_vec(9), {})}
    store = _store("acme", idx)
    res = sync_delta(
        store,
        vec_by_slug={"a": _vec(1), "b": _vec(2)},
        meta_by_slug={"a": {"kind": "page"}, "b": {"kind": "chunk"}},
        upsert_slugs={"a", "b"},
        delete_ids=[],
        reset=True,
    )
    assert res["reset"] is True and res["upserted"] == 2
    ns = idx.namespaces["acme"]
    assert set(ns) == {"a", "b"}  # "old" was cleared by reset


def test_sync_delta_incremental_upserts_delta_and_deletes_removed():
    idx = FakeIndex()
    idx.namespaces["acme"] = {"a": (_vec(1), {}), "c": (_vec(3), {})}
    store = _store("acme", idx)
    res = sync_delta(
        store,
        vec_by_slug={"a": _vec(1), "b": _vec(2)},
        meta_by_slug={"a": {}, "b": {}},
        upsert_slugs={"b"},        # only the changed/new one
        delete_ids=["c"],          # removed
        reset=False,
    )
    assert res["reset"] is False and res["upserted"] == 1 and res["deleted"] == 1
    ns = idx.namespaces["acme"]
    assert set(ns) == {"a", "b"}  # a untouched, b added, c deleted


def test_sync_delta_noop_when_disabled():
    store = TenantStore("company-wiki", "acme", index=FakeIndex())
    store.enabled = False  # simulate no Pinecone key
    res = sync_delta(store, vec_by_slug={}, meta_by_slug={}, upsert_slugs=set(), delete_ids=[], reset=True)
    assert res == {"enabled": False}
