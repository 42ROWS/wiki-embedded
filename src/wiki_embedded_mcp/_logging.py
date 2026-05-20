"""Structured logger for wiki-embedded-mcp.

Single entrypoint (`get_logger`) so all modules share the same logger hierarchy.
Level resolved from `WIKI_EMBEDDED_LOG_LEVEL` env var (default INFO).
Logs go to stderr — MCP servers must keep stdout reserved for the JSON-RPC stream.
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT_LOGGER_NAME = "wiki_embedded_mcp"
_INITIALIZED = False


def _init() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    level_name = os.getenv("WIKI_EMBEDDED_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the wiki_embedded_mcp namespace."""
    _init()
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
