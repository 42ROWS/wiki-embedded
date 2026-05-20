# polars-runner

> Natural-language → [Polars](https://pola.rs) code, with auto-improving RAG cache. Runs on **Apify**, **Docker**, or as a **Python library**.

Part of the [**wiki-embedded**](https://github.com/42ROWS/wiki-embedded) monorepo (`packages/polars-runner/`).

## What it does

You describe a data transformation in plain English, the runner generates Polars code with an LLM, executes it in a sandbox, and returns the results — plus the code itself for inspection and reuse.

- **Multi-format input**: CSV, JSON, Excel, Parquet, or a JSON object passed inline.
- **Multi-table mode**: ship `{ "contacts": [...], "companies": [...] }` and the LLM gets one schema per table for explicit JOINs.
- **Three tiers**:
  - **BYOK** — bring your Google / Anthropic / OpenAI / Groq key.
  - **Hosted basic** — Gemini 2.5 Flash-Lite (cheap, fast).
  - **Hosted premium** — Gemini 2.5 Pro with reasoning (slow, high quality).
- **Auto-improving**: every successful execution is embedded and stored in a Pinecone index; future similar prompts retrieve and reuse working code, skipping the generation step.
- **Error recovery**: on a sandbox exception, the analyzer extracts a structured hint (missing column, wrong dtype, JOIN mismatch) and asks the LLM to fix-and-retry up to N times.

## Two build targets

```
packages/polars-runner/
├── .actor/Dockerfile           Apify deploy  (apify push)
└── builds/docker/Dockerfile    Standalone   (docker build)
```

Same source, different wrappers — see the `Dockerfile` and `docker-compose.yml` files for runtime knobs.

## Quick start

```bash
# Install (workspace member of wiki-embedded)
uv sync --package polars-runner

# Run with your own key, single table
GOOGLE_API_KEY=... polars-runner \
    --input data.csv \
    --prompt "Group by sector, sum revenue, top 10 desc"
```

## Why it lives here

It is the analytics engine that backfills `analytics/` chunks at compile time for [`wiki-compiler`](../wiki-compiler/), and may be called at runtime by [`wiki-embedded-mcp`](../../) for ad-hoc graph queries when a precomputed answer is not available. Standalone use is fully supported.

## Status

`v0.3.0` — refactored into the monorepo, FAANG-grade packaging. The post-execution **semantic validator** (planned `v0.4`) will tackle the "status SUCCEEDED but answer is semantically wrong" class of bugs surfaced by our benchmark.

## License

MIT — see [LICENSE](../../LICENSE).
