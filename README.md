# wiki-embedded

[![PyPI](https://img.shields.io/pypi/v/wiki-embedded-mcp.svg)](https://pypi.org/project/wiki-embedded-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/wiki-embedded-mcp.svg)](https://pypi.org/project/wiki-embedded-mcp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-2496ED.svg)](https://github.com/42ROWS/wiki-embedded/pkgs/container/wiki-embedded-mcp)
[![MCP](https://img.shields.io/badge/MCP-server-7c3aed.svg)](https://modelcontextprotocol.io/)

**Compile any markdown wiki into a knowledge base your AI agent can query as if it were an API.** A single `.zip` bundle, plug-and-play in Claude Desktop / Cursor / Cline via the [Model Context Protocol](https://modelcontextprotocol.io/). Cross-lingual semantic search out of the box, **5× fewer tokens** per question, optional analytics with a [property-based testing oracle](#the-oracle-property-based-testing-for-llm-generated-code).

> **In one sentence** — Turn 1851 markdown files into a 5.4 MB bundle. Ask questions in any language. Get answers with citations in ~2 seconds instead of feeding the whole wiki to your LLM every time.

---

## The 3 components

```
   ┌──────────────────────────────┐
   │  wiki-compiler                │   Apify actor + Docker
   │  Markdown wiki → chunks       │   • parse + crossref
   │                + embeddings   │   • LLM-teacher answer chunks
   │                + manifest     │   • E5 multilingual encoder
   └────────────┬─────────────────┘
                │ compiled_wiki.zip (~5 MB)
                ▼
   ┌──────────────────────────────┐
   │  wiki-embedded-mcp            │   pip / Docker / ghcr
   │  MCP server                   │   • served via stdio JSON-RPC
   │                               │   • cosine retrieval on E5 vectors
   │                               │   • backlinks / graph tools
   └──────────────────────────────┘
                │ MCP protocol
                ▼
       Claude Desktop / Cursor / Cline

  + optional side-channel for analytics:
   ┌──────────────────────────────┐
   │  polars-runner                │   Apify actor + Docker
   │  NL → Polars code             │   • Gemini 2.5 Pro derives
   │                               │     PBT oracle from prompt
   │                               │   • deterministic verifier
   │                               │   • RAG only saves what passes
   └──────────────────────────────┘
```

---

## Benchmark — measured, not estimated

20 strategic queries (Italian + English mix) on a real 1851-page wiki, comparing:
- **wiki-embedded** (this project): top-5 E5-cosine retrieval, then Gemini 2.5 Flash for answer synthesis
- **filesystem baseline**: `grep -rli` on the raw markdown directory, then Gemini 2.5 Flash on every matched file (capped at 20)

| Metric | wiki-embedded | filesystem grep | Winner |
|---|---|---|---|
| Retrieval latency | 322 ms median | 69 ms median | grep (Pinecone Inference round-trip costs ~250 ms) |
| **Input tokens to LLM** | **2 035 mean** | **9 255 mean** | **wiki-embedded — 78% saving** |
| End-to-end latency (estimate at LLM speed) | ~2.3 s | ~9.1 s | **wiki-embedded ~4× faster** |
| Answer quality (LLM-as-judge, 8 valid samples) | 5 wins | 3 wins | wiki-embedded |

The retrieval step alone is faster for grep — but **the LLM call is the bottleneck**, and feeding 5× more text makes the LLM slower *and* more expensive *and*, in our test, less accurate (it gets lost in noise and replies "not enough information").

Two queries where the difference was obvious:

**Q12 — "Sintesi del segmento Professional Services & Legal Compliance"**
- *wiki-embedded* → full structured paragraph naming the segment's clients, tools (TeamSystem, Wolters Kluwer), and use cases, citing `[[segments/professional-services-legal-compliance]]`
- *grep* → "The passages do not contain enough information…"

**Q16 — "Initiative con funnel stage MOFU e ad_format single_image"**
- *wiki-embedded* → cites two specific initiative pages by slug
- *grep* → "The provided passages do not contain enough information…"

Full results, including the input/output tokens per query, are in [`benchmarks/retrieval_results.json`](./benchmarks/retrieval_results.json).

---

## Quick start

### Option A — pre-compiled bundle (recommended)

1. Compile your wiki once with the [Apify actor](https://apify.com/salesmart-srl/42rows-wiki-compiler), or self-host the compiler (`packages/wiki-compiler/`). The output is one `compiled_wiki.zip`.

2. Run the MCP server:

```bash
pip install 'wiki-embedded-mcp[cloud]'
export PINECONE_API_KEY="..."   # required when the bundle uses pinecone: embeddings
wiki-embedded-mcp --compiled-wiki https://.../compiled_wiki.zip
```

3. Wire it into your agent:

```json
{
  "mcpServers": {
    "wiki-embedded": {
      "command": "wiki-embedded-mcp",
      "args": ["--compiled-wiki", "https://.../compiled_wiki.zip"],
      "env": { "PINECONE_API_KEY": "..." }
    }
  }
}
```

### Option B — local wiki directory (compile on first boot)

```bash
pip install 'wiki-embedded-mcp[all]'
wiki-embedded-mcp --wiki ./my-wiki   # E5 model downloaded on first run
```

### Option C — Docker

```bash
# Pre-built image (lightweight, cloud embeddings)
docker pull ghcr.io/42rows/wiki-embedded-mcp:latest

# Full self-contained (bundles sentence-transformers + torch)
docker pull ghcr.io/42rows/wiki-embedded-mcp:latest-full
```

---

## MCP tools exposed

When connected, the server exposes the following tools to your AI agent:

| Tool | Purpose |
|---|---|
| `query_wiki(question, top_k=5)` | semantic search, returns top-K passages with `[[slug]]` citations |
| `get_page(slug)` | full markdown of a single page |
| `get_backlinks(slug)` | which pages cite a given page (reverse crossref) |
| `get_graph(slug, depth=1)` | local subgraph around a page |
| `list_thesis()` | the auto-derived wiki "lens" (target audience, key concepts, excluded topics) |

---

## The oracle — Property-Based Testing for LLM-generated code

The `polars-runner` package is an optional **analytics side-channel**: ask data questions in natural language ("which pages cite the most authors?", "show me the orphan companies"), get verified Polars code + result.

The novelty is the **PBT oracle**:

1. **Gemini 2.5 Pro extracts a property contract** from your prompt — universal properties any valid answer must satisfy (expected row range, required columns, non-null fields, value ranges, sort order). One LLM call per prompt, ~$0.02.
2. **A basic-tier model generates Polars code** and executes it in a sandbox.
3. **A deterministic Python verifier** checks the resulting DataFrame against the oracle. No magic vocabulary, no language assumptions — the rules come from the prompt itself.
4. **The retry loop is triggered with the verifier's feedback** if the code produces a result that fails the contract (e.g. `group_by` on the wrong column collapses to 672 rows when the prompt clearly implied ≤ 5).
5. **The RAG skill-library only stores code that passed the oracle** — no more self-poisoning.

Pattern references:
- Vikram et al., *"Can Large Language Models Write Good Property-Based Tests?"*, [arxiv 2307.04346](https://arxiv.org/abs/2307.04346).
- Wang et al., *"From Prompts to Properties: Rethinking LLM Code Generation with Property-Based Testing"*, FSE 2024, [doi 10.1145/3696630.3728702](https://doi.org/10.1145/3696630.3728702).
- *"Use Property-Based Testing to Bridge LLM Code Generation and Validation"*, [arxiv 2506.18315](https://arxiv.org/abs/2506.18315).
- Zhang et al., *"ALGO: Synthesizing Algorithmic Programs with LLM-Generated Oracle Verifiers"*, ICLR 2024.

Shinn et al., *"Reflexion: Language Agents with Verbal Reinforcement Learning"*, [arxiv 2303.11366](https://arxiv.org/abs/2303.11366), is the precedent for using verifier feedback as the retry signal.

---

## When to use this (and when not to)

**Use it when**:
- Your wiki has 100+ files and you do not want to feed it all to your LLM every time
- You use Claude Desktop / Cursor / Cline / any MCP client — they don't see your local filesystem
- You want one portable `.zip` bundle to ship the wiki to colleagues or to production
- Your wiki is multilingual or your queries are
- You want analytics on top of the wiki without hand-writing Polars

**Skip it when**:
- Your wiki is ≤ 30 files and you are the only reader
- You already work in Claude Code or another IDE that grep-reads the filesystem
- You publish your wiki for human readers, not as a knowledge source for agents

---

## Architecture

```
mario-wiki-mcp/                          ← monorepo (uv workspace)
├── src/wiki_embedded_mcp/               ← the MCP server (pip + Docker)
├── packages/
│   ├── wiki-compiler/                   ← Apify actor + standalone Docker
│   └── polars-runner/                   ← Apify actor + standalone Docker
├── scripts/                             ← operations (Pinecone audit & purge)
├── benchmarks/                          ← retrieval bench harness + results
└── pyproject.toml                       ← uv workspace root
```

Each `packages/*/` is independently publishable. The MCP server reads a `compiled_wiki.zip` from any URL or local path; the compiler emits one; the polars-runner is consumed by the MCP server's optional `query_analytics` tool.

---

## Comparison vs alternatives

| | wiki-embedded | DeepWiki | LangChain RAG | Karpathy LLM Wiki (gist) |
|---|---|---|---|---|
| MCP-native | ✓ | partial | needs glue | ✗ |
| Self-hosted | ✓ | ✗ | ✓ | ✓ |
| Pre-computed answer chunks | ✓ (LLM teacher) | ✗ | ✗ | ✓ |
| One-zip portable bundle | ✓ | ✗ | ✗ | partial |
| Cross-language semantic | ✓ (E5 multilingual) | partial | depends on embedder | ✗ |
| Property-based testing oracle (analytics) | ✓ | ✗ | ✗ | ✗ |
| LLM call per question | only synthesis | yes | yes | yes |

---

## Roadmap

- [x] v0.1: MCP server scaffold + compiled_wiki bundle loader
- [x] v0.2: wiki-compiler Apify actor, multi-lingual E5 embeddings
- [x] v0.3: PBT oracle in polars-runner, monorepo unification, benchmark harness
- [ ] v0.4: tighter oracle instruction (force per-X cardinality), pytest coverage ≥ 80%
- [ ] v0.5: thesis-conditional retrieval (rerank by purpose lens), analytics chunks compile-time
- [ ] v1.0: stable Python API, semantic versioning, frozen MCP tool schema

---

## License

MIT — see [LICENSE](./LICENSE).

Authored by [Mario Brosco](mailto:info@42rows.com), maintained by [42rows S.r.l.](https://42rows.com).
