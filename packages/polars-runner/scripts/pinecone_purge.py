#!/usr/bin/env python3
"""Selective purge of polars-runner Pinecone vectors.

The polars-runner learns from past transformations by storing successful code
in the ``code-success`` namespace. Before the property-based oracle existed,
"successful" only meant "produced rows", so syntactically valid but
semantically wrong code (e.g. wrong GROUP BY key) could be saved with a
quality score that subsequent retrievals trusted. This script removes such
legacy vectors so the oracle-gated pipeline starts from a clean slate.

The default behaviour is **dry-run**. Pass ``--apply`` to actually delete.

Usage:
    # Inspect what would be removed (no deletion):
    PINECONE_API_KEY=... python -m scripts.pinecone_purge \\
        --namespace code-success \\
        --min-quality 90 \\
        --validator-passed-only

    # Apply the same purge:
    PINECONE_API_KEY=... python -m scripts.pinecone_purge \\
        --namespace code-success --min-quality 90 \\
        --validator-passed-only --apply

    # Nuclear option (every vector in the namespace):
    PINECONE_API_KEY=... python -m scripts.pinecone_purge \\
        --namespace code-success --all --apply --yes-i-am-sure
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from polars_runner.core.constants import RAG_CONFIG
from polars_runner.rag.pinecone_client import PineconeClient


_PAGE_SIZE = 1_000  # Pinecone allows up to ~10K per query; 1K keeps it cheap.


def _iter_matching_ids(
    index,
    namespace: str,
    dim: int,
    *,
    min_quality: float | None,
    require_validator_passed: bool,
    page_limit: int,
) -> list[str]:
    """Collect IDs of vectors that match the filter.

    Implemented as a single similarity query against a zero-probe. Pinecone
    has no native list-by-metadata cursor on the free tier, so we rely on
    top-k retrieval. ``page_limit`` bounds the number of vectors we will
    consider.
    """
    probe = [0.0] * dim
    response = index.query(
        namespace=namespace,
        vector=probe,
        top_k=page_limit,
        include_metadata=True,
    )
    ids: list[str] = []
    for match in response.matches:
        meta = match.metadata or {}
        quality = meta.get("quality_score")
        passed = meta.get("validator_passed") is True

        if min_quality is not None:
            if quality is None or quality < min_quality:
                continue
        if require_validator_passed and not passed:
            continue
        ids.append(match.id)
    return ids


def _delete_in_batches(index, namespace: str, ids: list[str], batch: int = 100) -> None:
    """Delete IDs in chunks to stay under Pinecone's per-call payload cap."""
    for offset in range(0, len(ids), batch):
        chunk = ids[offset : offset + batch]
        index.delete(ids=chunk, namespace=namespace)
        # Tiny pause to be polite with the free tier.
        time.sleep(0.05)


def main() -> int:
    p = argparse.ArgumentParser(description="Selective purge of Pinecone vectors.")
    p.add_argument(
        "--namespace",
        required=True,
        help="Pinecone namespace to operate on (e.g. 'code-success').",
    )
    p.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="Delete vectors whose `quality_score` metadata is >= this value.",
    )
    p.add_argument(
        "--validator-passed-only",
        action="store_true",
        help="Additionally require `validator_passed=True` in metadata.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Match every vector in the namespace (overrides other filters).",
    )
    p.add_argument(
        "--page-limit",
        type=int,
        default=10_000,
        help="Maximum vectors to consider in one pass (Pinecone top-k limit).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Without this flag the script runs as dry-run and prints what it would delete.",
    )
    p.add_argument(
        "--yes-i-am-sure",
        action="store_true",
        help="Required together with --all --apply to confirm a full namespace wipe.",
    )
    args = p.parse_args()

    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY env var not set.", file=sys.stderr)
        return 2

    if args.all and args.apply and not args.yes_i_am_sure:
        print("ERROR: --all --apply requires --yes-i-am-sure", file=sys.stderr)
        return 2

    client = PineconeClient()
    index = client.get_index()
    dim = RAG_CONFIG.embedding_dimension

    stats = index.describe_index_stats()
    ns_stats = stats.namespaces.get(args.namespace, {})
    total = ns_stats.get("vector_count", 0)
    print(f"Namespace '{args.namespace}': {total} vector(s) total\n")

    if total == 0:
        print("Nothing to do.")
        return 0

    if args.all:
        # No filter — collect everything we can see.
        ids = _iter_matching_ids(
            index,
            args.namespace,
            dim,
            min_quality=None,
            require_validator_passed=False,
            page_limit=min(args.page_limit, total),
        )
    else:
        ids = _iter_matching_ids(
            index,
            args.namespace,
            dim,
            min_quality=args.min_quality,
            require_validator_passed=args.validator_passed_only,
            page_limit=args.page_limit,
        )

    print(f"Matched {len(ids)} vector(s) under the filter.")
    if not ids:
        return 0

    if not args.apply:
        print("(dry-run — pass --apply to delete)")
        for vid in ids[:10]:
            print(f"  would delete: {vid}")
        if len(ids) > 10:
            print(f"  … and {len(ids) - 10} more")
        return 0

    print(f"Deleting {len(ids)} vector(s) in batches…")
    _delete_in_batches(index, args.namespace, ids)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
