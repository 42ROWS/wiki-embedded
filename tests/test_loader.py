"""Tests for loader.load_compiled_wiki — bundle parsing + validation."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from wiki_embedded_mcp.loader import (
    CompiledWikiError,
    _validate_manifest,
    load_compiled_wiki,
)


def test_load_valid_bundle(sample_compiled_bundle: Path) -> None:
    loaded = load_compiled_wiki(sample_compiled_bundle)
    assert loaded.embeddings.shape == (5, 4)
    assert len(loaded.slug_order) == 5
    assert loaded.embedding_model == "test:tiny-4d"
    assert loaded.embedding_dims == 4
    assert loaded.manifest["compiler"] == "test-fixture"
    assert loaded.thesis_text  # not empty


def test_load_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_compiled_wiki("/nonexistent/path/compiled.zip")


def test_load_corrupt_zip_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(CompiledWikiError):
        load_compiled_wiki(bad)


def test_load_missing_manifest_raises(tmp_path: Path) -> None:
    bundle = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("pages/x.md", "---\nslug: pages/x\n---\nbody")
    with pytest.raises(CompiledWikiError, match="manifest.json missing"):
        load_compiled_wiki(bundle)


def test_load_missing_embeddings_raises(tmp_path: Path) -> None:
    bundle = tmp_path / "no_emb.zip"
    manifest = {
        "format_version": 1,
        "compiler": "test",
        "wiki": {"pages_count": 0},
        "embeddings": {"model": "x", "dims": 4},
    }
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("pages/x.md", "---\nslug: pages/x\n---\nbody")
    with pytest.raises(CompiledWikiError, match="embeddings.npz missing"):
        load_compiled_wiki(bundle)


def test_validate_manifest_unsupported_version() -> None:
    with pytest.raises(CompiledWikiError, match="unsupported format_version"):
        _validate_manifest({
            "format_version": 99,
            "compiler": "x", "wiki": {},
            "embeddings": {"model": "y", "dims": 4},
        })


def test_validate_manifest_missing_field() -> None:
    with pytest.raises(CompiledWikiError, match="missing required fields"):
        _validate_manifest({"format_version": 1})
