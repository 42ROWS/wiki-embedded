"""polars-runner — natural-language to Polars code, with a property-based
oracle quality gate and an auto-improving RAG skill library.

Importing this package is deliberately side-effect free: heavy modules
(``polars``, ``apify``, ``google-genai``, ``pinecone``) are not loaded until
you reach into the relevant submodule. This lets lightweight tooling under
``scripts/`` (auditing, purging the RAG store, etc.) work without pulling
the full runtime stack.
"""
from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
