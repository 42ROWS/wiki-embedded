"""Wiki Embedded MCP — Persistent knowledge for AI agents.

Karpathy LLM Wiki + materialized views (pre-computed answer chunks) + thesis-conditioning.
Drop-in fast retrieval for LLM agents via the Model Context Protocol (MCP).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("wiki-embedded-mcp")
except PackageNotFoundError:  # editable install / source tree
    __version__ = "0.0.0+local"


__all__ = ["__version__"]
