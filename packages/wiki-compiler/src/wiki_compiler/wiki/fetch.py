"""Fetch markdown sources into a local temp directory before compilation.

Supports 3 modes (matching .actor/input_schema.json `wikiSource`):
- github: shallow-clone a public repo (optionally a subpath)
- zipUrl: HTTP GET + unzip
- uploadedFiles: copy from Apify KV-store / local paths into a flat dir
"""
from __future__ import annotations
import io
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass
class FetchedWiki:
    local_dir: Path
    source_kind: str
    origin: str  # human-readable origin (repo URL, zip URL, "uploaded:N files")


def fetch_github(repo: str, branch: str = "main") -> FetchedWiki:
    """Fetch a public GitHub repo (or subpath) via codeload tarball.

    `repo` can be `owner/repo` or `owner/repo/subpath/to/wiki`.
    """
    parts = repo.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid githubRepo: {repo!r} — expected owner/repo[/subpath]")
    owner, name = parts[0], parts[1]
    subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

    url = f"https://codeload.github.com/{owner}/{name}/zip/refs/heads/{branch}"
    with httpx.Client(timeout=60.0, follow_redirects=True) as cli:
        r = cli.get(url)
        r.raise_for_status()

    tmp = Path(tempfile.mkdtemp(prefix="wiki-gh-"))
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(tmp)

    # zip root is `<name>-<branch>/`
    extracted_root = next(tmp.iterdir())
    target = extracted_root / subpath if subpath else extracted_root

    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(
            f"Subpath {subpath!r} not found in {owner}/{name}@{branch}"
        )

    return FetchedWiki(local_dir=target, source_kind="github", origin=f"{owner}/{name}@{branch}/{subpath}")


def fetch_zip_url(url: str) -> FetchedWiki:
    """Download a public ZIP and extract it."""
    with httpx.Client(timeout=120.0, follow_redirects=True) as cli:
        r = cli.get(url)
        r.raise_for_status()

    tmp = Path(tempfile.mkdtemp(prefix="wiki-zip-"))
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(tmp)

    return FetchedWiki(local_dir=tmp, source_kind="zipUrl", origin=url)


def stage_uploaded_files(file_paths: list[Path]) -> FetchedWiki:
    """Copy uploaded `.md` files into a flat staging dir."""
    tmp = Path(tempfile.mkdtemp(prefix="wiki-upload-"))
    n = 0
    for src in file_paths:
        src = Path(src)
        if not src.is_file():
            continue
        if src.suffix.lower() != ".md":
            continue
        shutil.copy2(src, tmp / src.name)
        n += 1
    return FetchedWiki(local_dir=tmp, source_kind="uploadedFiles", origin=f"uploaded:{n} files")
