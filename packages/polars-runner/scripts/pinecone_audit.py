#!/usr/bin/env python3
"""Audit the polars-runner Pinecone index.

Prints summary statistics for each namespace (vector count, distribution of
``quality_score``, share of ``validator_passed=True``, recent activity).
Useful before running :mod:`pinecone_purge` to understand what would be
removed.

Usage:
    PINECONE_API_KEY=... python -m scripts.pinecone_audit
    PINECONE_API_KEY=... python -m scripts.pinecone_audit --namespace code-success
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# Make the package importable when run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from polars_runner.core.constants import RAG_CONFIG
from polars_runner.rag.pinecone_client import PineconeClient


# Maximum vectors we sample when computing distributions. Pinecone has no
# bulk-fetch API; we approximate by querying with a zero-vector at high
# top_k. 10_000 is well under both Pinecone's per-query cap and our index
# capacity.
_SAMPLE_TOP_K = 10_000


def _bucketize_quality(scores: list[float]) -> dict[str, int]:
    buckets = Counter()
    for s in scores:
        if s is None:
            buckets["unset"] += 1
        elif s >= 90:
            buckets["90-100"] += 1
        elif s >= 75:
            buckets["75-89"] += 1
        elif s >= 50:
            buckets["50-74"] += 1
        elif s >= 25:
            buckets["25-49"] += 1
        else:
            buckets["0-24"] += 1
    return dict(buckets)


def _audit_namespace(index, namespace: str, dim: int) -> dict[str, object]:
    stats = index.describe_index_stats()
    ns_stats = stats.namespaces.get(namespace)
    if not ns_stats or ns_stats.get("vector_count", 0) == 0:
        return {"namespace": namespace, "vector_count": 0, "note": "empty"}

    total = ns_stats["vector_count"]

    # Use a zero-vector as a "random" probe to fetch up to _SAMPLE_TOP_K
    # records along with their metadata. Pinecone returns them ranked by
    # similarity to the probe — for an audit that ordering is irrelevant.
    probe = [0.0] * dim
    sample = index.query(
        namespace=namespace,
        vector=probe,
        top_k=min(_SAMPLE_TOP_K, total),
        include_metadata=True,
    )
    matches = list(sample.matches)
    qualities = [m.metadata.get("quality_score") for m in matches]
    validator_pass = sum(1 for m in matches if m.metadata.get("validator_passed") is True)
    has_quality_100 = sum(1 for q in qualities if q is not None and q >= 99.5)

    return {
        "namespace": namespace,
        "vector_count_total": total,
        "vectors_sampled": len(matches),
        "quality_distribution": _bucketize_quality(qualities),
        "validator_passed_share": (
            f"{validator_pass}/{len(matches)} "
            f"({100 * validator_pass / max(1, len(matches)):.1f}%)"
        ),
        "quality_100_count": has_quality_100,
        "median_quality": (
            round(sorted(q for q in qualities if q is not None)[len(qualities) // 2], 1)
            if qualities and any(q is not None for q in qualities)
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a polars-runner Pinecone namespace.")
    parser.add_argument(
        "--namespace",
        action="append",
        help="Namespace to audit (repeatable). Defaults to all configured namespaces.",
    )
    args = parser.parse_args()

    if not os.getenv("PINECONE_API_KEY"):
        print("ERROR: PINECONE_API_KEY env var not set.", file=sys.stderr)
        return 2

    namespaces = args.namespace or [
        RAG_CONFIG.namespace_success,
        RAG_CONFIG.namespace_failures,
    ]

    client = PineconeClient()
    index = client.get_index()
    dim = RAG_CONFIG.embedding_dimension

    print(f"Index: {client._index_name}  dim={dim}\n")  # noqa: SLF001
    for ns in namespaces:
        report = _audit_namespace(index, ns, dim)
        print(f"━━━ namespace: {report['namespace']} ━━━")
        for key, value in report.items():
            if key == "namespace":
                continue
            print(f"  {key:30} {value}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
