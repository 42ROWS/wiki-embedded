# Contributing to Wiki Embedded MCP

Thanks for taking the time to contribute! This package is part of the [42rows wiki-embedded monorepo](https://github.com/42ROWS/wiki-embedded).

## Quick setup

```bash
git clone https://github.com/42ROWS/wiki-embedded.git
cd wiki-embedded/packages/wiki-embedded-mcp
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
pytest
```

## Project layout

```
src/wiki_embedded_mcp/
├── server.py       # MCP entrypoint + tool/resource/prompt handlers
├── compile.py      # parse markdown wiki → pages + crossref graph
├── loader.py       # load a compiled_wiki.zip bundle
├── index.py        # in-memory retrieval + graph queries
├── embedder.py     # dual-backend (Pinecone | sentence-transformers)
├── resources.py    # MCP Resources URI scheme (wiki-embedded://slug)
├── prompts.py      # MCP Prompts templates
└── _logging.py     # logger setup
tests/              # pytest suite
```

## Pull requests

1. Fork + branch off `main`.
2. Run `ruff check src tests && ruff format --check src tests`.
3. Run `pytest`.
4. Commit messages: imperative (`fix: handle empty wiki dir`), reference issue if any (`Closes #42`).
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open the PR; small, focused PRs are merged faster.

## Adding a new MCP tool

1. Declare the `Tool` in `server.py:TOOLS` with a precise `description` and JSON Schema.
2. Add the handler in the `call_tool` dispatch + a `_my_tool(...)` impl function.
3. Add a test in `tests/test_server.py`.
4. Document in `README.md` under "Tool surface".

## Adding a new embedding backend

Implement the `Embedder` interface in `embedder.py` by adding a third branch (`pinecone` | `local` | `<new>`) and route at construction via the model-name prefix.

## Releasing

Maintainers only. Bump version in `pyproject.toml`, tag `vX.Y.Z`, GitHub Actions builds + publishes to PyPI and GHCR.

## Code of Conduct

Be excellent to each other. See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).
