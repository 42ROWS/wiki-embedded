# Changelog

All notable changes to `wiki-embedded-mcp` are documented here. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Vertex AI support for Gemini** — set `GOOGLE_GENAI_USE_VERTEXAI=true`
  (+ `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, service-account creds) to
  route thesis + chunk generation through Vertex AI instead of the AI Studio
  `api_key`. Required where the AI Studio API is geo-restricted. New helper
  `wiki_compiler.wiki._google.genai_client`; no api_key needed in Vertex mode.
- **Per-(collection, tenant) Pinecone retrieval store** — opt-in via `tenantId`
  (+ `collection`, default `company-wiki`). Maps to Pinecone natively:
  `collection → index`, `tenant → namespace`. On compile the vectors are synced
  to that namespace; an agent then retrieves with `TenantStore(collection,
  tenant).search(query, embedder)`. The handle is **bound** to one
  `(collection, tenant)` at construction, so read/write isolation is by
  construction. The incremental diff drives the sync 1:1 (upsert added/changed/
  regenerated, delete removed, no-op carried); a full rebuild resets the
  namespace then upserts all. No-op (bundle only) when `PINECONE_API_KEY` is
  absent or `tenantId` is empty. New module `wiki_compiler.tenant_store`.
- **Incremental update** for `wiki-compiler` — the compiled bundle is now a
  diff baseline (like a lockfile). Pass a prior `compiled_wiki.zip` back via
  `priorBundleUrl` (actor) / `priorBundlePath` (local) and the compiler
  re-embeds only added/changed pages and regenerates only chunks whose evidence
  moved, reusing everything else bit-identically. Typical small edits are
  ~5–10× faster/cheaper than a full recompile.
  - New `state.json` member inside the bundle: the record manager
    (`{slug → content_hash}` for pages, `{archetype_id → {hash, evidence_slugs,
    source_query}}` for chunks, plus the cached archetype set, thesis hash, and
    embedding model/dims). Written on every compile (full or incremental).
  - Page change detection hashes `content_full` (the exact text embedded), so a
    frontmatter-only edit does not trigger a re-embed.
  - Chunk staleness = evidence touched a changed/removed page **or** the
    recomputed top-k evidence drifted (cheap cosine re-retrieval, no LLM).
  - Guards: a prior bundle with a different embedding model, no `state.json`
    (pre-feature bundle), or a changed fraction above the threshold falls back
    to a full rebuild automatically — never worse than a full compile.
  - Modules: `wiki_compiler.state`, `wiki_compiler.incremental`;
    `make_archetype_id()` exposed for stable chunk identity.

## [0.3.0] — 2026-05-18

Structural pivot. The repository is now a **uv workspace with three
independently publishable packages**, the `polars-runner` gains a
**property-based testing oracle** as its quality gate, and the project
ships its first end-to-end **retrieval benchmark** against the
filesystem-grep baseline.

### Added
- **`packages/polars-runner/`** — natural-language to Polars code, with a
  property-based testing oracle as quality gate. Gemini 2.5 Pro extracts
  a property contract from each user prompt (expected rows range,
  required columns, non-null fields, value ranges, sort order); a
  deterministic Python verifier checks the result frame and triggers the
  retry loop with the verifier's feedback. The auto-improving RAG
  skill-library only accepts code that passes the oracle. Backed by the
  FSE 2024 *"From Prompts to Properties"* paper and Voyager-style skill
  curation (NeurIPS 2023).
- **`packages/wiki-compiler/`** — Apify actor + standalone Docker target
  for the compile-time pipeline (parse markdown → derive thesis → emit
  pre-computed answer chunks → embed → ship `compiled_wiki.zip`).
- **`scripts/pinecone_audit.py` + `pinecone_purge.py`** — operations
  tools for inspecting and selectively cleaning the auto-improving RAG
  cache. Default is dry-run.
- **`benchmarks/retrieval_bench.py`** — 20-query harness comparing
  wiki-embedded vs filesystem grep on a 1851-page wiki, with LLM judge
  (Gemini 2.5 Flash via Vertex AI). Results in
  [`benchmarks/retrieval_results.json`](./benchmarks/retrieval_results.json).
- **Multi-table schema awareness** in `oracle.generate_oracle`: when the
  input is a dict of named tables, the oracle prompt is split into
  `# df_<name>` sections so the LLM sees real column names instead of
  guessing synonyms.

### Changed
- **The oracle is now the default quality gate**, not an optional
  feature. Disable explicitly with `POLARS_RUNNER_DISABLE_ORACLE=1`.
  Old opt-in flags removed.
- **`semantic_validator.py` rewritten** to keep only the two universally
  safe checks (100% NULL columns on a non-empty frame; non-portable
  `is_in(<LazyFrame>)` idiom). All language-specific regex and
  domain-specific column blocklists were removed.
- **`RAGConfig.min_quality_for_retrieval`** is consumed at query time:
  vectors below the threshold are filtered out of similarity search.
- **Repository layout** moved to `packages/*` with implicit namespace
  packages and lazy package `__init__.py` so operations scripts can
  import metadata without pulling the full runtime stack (polars,
  apify SDK).

### Removed
- `executor/self_consistency.py` — the module had no caller.
- `is_llm_judge_enabled()` helper in `semantic_validator`.
- Unused parameters (`prompt`, `input_rows_total`, `api_key`,
  `use_llm_judge`) from `validate_semantic()`.
- 26 legacy poisoned vectors in the Pinecone `code-success` namespace,
  identified and purged with the new `pinecone_purge.py` tooling.

### Fixed
- Gemini 2.5 Pro 400 `INVALID_ARGUMENT` regression caused by
  `thinking_budget=0` in the oracle client (Pro requires thinking mode).
- Oracle empty-response failure when `oracle_max_tokens=1024` (Pro
  reasoning exhausts the budget before emitting JSON); default raised
  to **4096**.
- Hard-coded Apify token in `packages/wiki-compiler/tests/conftest.py`
  removed; integration tests now skip cleanly when `APIFY_TOKEN` is
  unset.

### Benchmarks
20 queries (Italian + English) on a 1851-page wiki:

| Metric | wiki-embedded | filesystem grep + LLM | Margin |
|---|---|---|---|
| Input tokens to LLM | 2 035 mean | 9 255 mean | **78% saving** |
| Retrieval latency | 322 ms median | 69 ms median | grep ~5× faster on retrieval alone |
| End-to-end latency (estimate at LLM speed) | ~2.3 s | ~9.1 s | **~4× faster** end-to-end |
| Judge wins (8 valid samples) | 5 | 3 | wiki-embedded |

## [0.2.0] — 2026-05-17

### Added
- **Compiled bundle loader** — new `--compiled-wiki <path|url>` flag and `WIKI_EMBEDDED_COMPILED` env var. Loads a `compiled_wiki.zip` produced by the [42rows Wiki Compiler](https://apify.com/salesmart-srl/42rows-wiki-compiler) Apify actor with instant boot (no re-embedding).
- **Dual-backend embedder** — model name prefix `pinecone:` routes to Pinecone Inference (cloud, fast); anything else uses `sentence-transformers` (local CPU).
- **MCP Resources** — every wiki page is exposed at `wiki-embedded://<slug>`, plus `wiki-embedded://thesis` and `wiki-embedded://__manifest__`.
- **MCP Prompts** — 3 ready-to-use templates: `summarize_wiki`, `ask_about`, `compare_topics`.
- **Graph tools** — `get_wiki_backlinks(slug)` and `get_wiki_graph(slug, depth)` for crossref navigation.
- **Structured logging** — all modules log to stderr via the `wiki_embedded_mcp` logger hierarchy. Level via `WIKI_EMBEDDED_LOG_LEVEL` env var.
- **Manifest validation** — `compiled_wiki.zip` bundles are sanity-checked before use; bad bundles fail fast with a clear error.
- **HTTP retry** — compiled-wiki download retries on transient network failures (3 attempts).
- **Tempdir cleanup** — extracted bundle dirs are removed on process exit (no filesystem leak).
- **Docker image** — official `ghcr.io/42rows/wiki-embedded-mcp:latest` for Claude Desktop / Cursor / Cline users who prefer containers over pip.
- **`--version` flag**, `__all__` exports, `py.typed` marker for type-aware IDEs.

### Changed
- Embedder errors now raise `EmbedderConfigError` with actionable install hints (e.g. `pip install 'wiki-embedded-mcp[local]'`).
- Optional dependencies split into `[local]`, `[cloud]`, `[all]` extras.

### Fixed
- `__init__.py` version now reads from installed package metadata (no more desync with `pyproject.toml`).
- Tool handlers wrap all exceptions and never crash the server.

## [0.1.0] — 2026-05-13

### Added
- Initial release: MCP server reading a local wiki directory with E5 embeddings.
- 7 tools: `query_wiki`, `query_wiki_answer`, `query_wiki_pages`, `read_wiki_page`, `list_wiki_pages`, `get_thesis`, `get_provenance`.
