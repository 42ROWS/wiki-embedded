"""Load a compiled_wiki.zip bundle produced by `42rows-wiki-compiler`.

Bundle layout (see actor `exporter.py`):
    pages/<slug>.md           # markdown sources
    chunks/<thesis_hash>/*.md # pre-computed answer chunks
    thesis.md                 # derived thesis (frontmatter + body)
    embeddings.npz            # numpy archive: vectors (float32, NxD) + slug_order
    manifest.json             # all metadata

`load_compiled_wiki` accepts an http(s) URL or a local path, downloads/extracts
into a tempdir, validates the bundle, and returns a `LoadedWiki` struct that the
MCP server uses directly (no re-embedding needed at runtime).

The tempdir is registered with `atexit` and cleaned up on process exit.
"""
from __future__ import annotations

import atexit
import io
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._logging import get_logger
from .compile import compile_wiki

log = get_logger("loader")

# Track tempdirs for atexit cleanup
_TEMPDIRS_TO_CLEANUP: list[Path] = []


def _register_tempdir(path: Path) -> None:
    _TEMPDIRS_TO_CLEANUP.append(path)


@atexit.register
def _cleanup_tempdirs() -> None:
    for p in _TEMPDIRS_TO_CLEANUP:
        try:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:  # never raise during interpreter shutdown
            pass


# Schema requirements for a compiled_wiki.zip manifest. Keep aligned with the
# actor `exporter.py:build_manifest()` output.
_REQUIRED_MANIFEST_FIELDS = ("format_version", "compiler", "wiki", "embeddings")
_SUPPORTED_FORMAT_VERSIONS = (1,)


@dataclass
class LoadedWiki:
    compiled: dict[str, Any]      # same shape as compile.compile_wiki output (dict)
    embeddings: np.ndarray        # (N, dims) float32
    slug_order: list[str]         # parallel to embeddings rows
    manifest: dict[str, Any]
    thesis_text: str
    thesis_meta: dict[str, Any]
    extract_dir: Path

    @property
    def embedding_model(self) -> str:
        return str(self.manifest.get("embeddings", {}).get("model", ""))

    @property
    def embedding_dims(self) -> int:
        return int(self.manifest.get("embeddings", {}).get("dims", 0))


class CompiledWikiError(Exception):
    """Raised when a compiled bundle is malformed or missing required artifacts."""


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _download(url: str, *, max_retries: int = 3, timeout_s: float = 120.0) -> bytes:
    """Download a URL with retry on transient HTTP errors."""
    import httpx

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            log.info("downloading compiled wiki (attempt %d/%d): %s", attempt, max_retries, url)
            with httpx.Client(timeout=timeout_s, follow_redirects=True) as cli:
                r = cli.get(url)
                r.raise_for_status()
                log.info("downloaded %d bytes", len(r.content))
                return r.content
        except (httpx.HTTPError, OSError) as e:
            last_exc = e
            log.warning("download attempt %d failed: %s", attempt, e)
    raise CompiledWikiError(f"failed to download {url} after {max_retries} attempts: {last_exc}")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Sanity-check the manifest before we trust its values."""
    missing = [f for f in _REQUIRED_MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise CompiledWikiError(f"manifest.json missing required fields: {missing}")
    fv = manifest.get("format_version")
    if fv not in _SUPPORTED_FORMAT_VERSIONS:
        raise CompiledWikiError(
            f"unsupported format_version={fv!r}, supported: {_SUPPORTED_FORMAT_VERSIONS}"
        )
    emb = manifest.get("embeddings", {})
    if not emb.get("model") or not emb.get("dims"):
        raise CompiledWikiError("manifest.embeddings must contain non-empty 'model' and 'dims'")


def load_compiled_wiki(source: str | Path) -> LoadedWiki:
    """Load a compiled_wiki.zip from URL or local path into memory.

    Raises:
        CompiledWikiError: bundle is malformed (missing files, bad manifest).
        FileNotFoundError: local path does not exist.
    """
    src = str(source)
    if _is_url(src):
        raw = _download(src)
    else:
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"compiled wiki not found: {p}")
        raw = p.read_bytes()
        log.info("loaded compiled wiki from local path: %s (%d bytes)", p, len(raw))

    extract = Path(tempfile.mkdtemp(prefix="wiki-embedded-compiled-"))
    _register_tempdir(extract)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(extract)
    except zipfile.BadZipFile as e:
        raise CompiledWikiError(f"compiled bundle is not a valid zip: {e}") from e

    # Manifest (validated first — fail fast on bad bundles)
    manifest_path = extract / "manifest.json"
    if not manifest_path.exists():
        raise CompiledWikiError(f"manifest.json missing inside bundle ({extract})")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise CompiledWikiError(f"manifest.json is not valid JSON: {e}") from e
    _validate_manifest(manifest)

    # Parse all markdown back into a compiled dict (pages + chunks merged)
    compiled_dc = compile_wiki(extract)
    compiled = compiled_dc if isinstance(compiled_dc, dict) else {
        "pages": compiled_dc.pages,
        "graph": compiled_dc.graph,
        "slug_set": compiled_dc.slug_set,
        "wiki_dir": str(extract),
    }

    # Embeddings
    npz_path = extract / "embeddings.npz"
    if not npz_path.exists():
        raise CompiledWikiError(f"embeddings.npz missing inside bundle ({extract})")
    try:
        npz = np.load(npz_path, allow_pickle=True)
        vectors = np.asarray(npz["vectors"], dtype=np.float32)
        slug_order = [str(s) for s in npz["slug_order"]]
    except (KeyError, ValueError) as e:
        raise CompiledWikiError(f"embeddings.npz malformed: {e}") from e

    if vectors.shape[0] != len(slug_order):
        raise CompiledWikiError(
            f"embeddings rows ({vectors.shape[0]}) != slug_order length ({len(slug_order)})"
        )

    # Thesis (optional)
    thesis_text = ""
    thesis_meta: dict[str, Any] = {}
    thesis_md = extract / "thesis.md"
    if thesis_md.exists():
        import frontmatter
        post = frontmatter.load(thesis_md)
        thesis_text = post.content
        thesis_meta = dict(post.metadata)

    log.info(
        "loaded compiled wiki: %d pages, %d embeddings, model=%s, dims=%d",
        len(compiled["pages"]), vectors.shape[0],
        manifest["embeddings"].get("model"), manifest["embeddings"].get("dims"),
    )

    return LoadedWiki(
        compiled=compiled,
        embeddings=vectors,
        slug_order=slug_order,
        manifest=manifest,
        thesis_text=thesis_text,
        thesis_meta=thesis_meta,
        extract_dir=extract,
    )
