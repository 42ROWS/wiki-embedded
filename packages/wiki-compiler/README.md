# 42rows Wiki Compiler

**Persistent knowledge wikis for AI agents** — turn any markdown wiki into a compiled, AI-agent-ready knowledge base. Auto-derives a *thesis*, pre-computes answer chunks via an LLM teacher, embeds with E5, and ships a single `compiled_wiki.zip` + MCP manifest you can plug into Claude Desktop, Cursor, or Cline.

```
Pre-computed answer chunks return at embedder speed (no per-query LLM);
incremental updates re-embed only what changed.
```

> Open-source (MIT). A hosted, managed version — with enrichment & campaign tooling — is available at **[42rows.com](https://42rows.com)**.

## What it does

1. **Fetches** your wiki: public GitHub repo (with optional subpath), public ZIP URL, or direct file upload.
2. **Compiles** all `.md` files into a normalized page set + crossref graph (`[[wikilinks]]`).
3. **Derives a thesis** (target audience, primary use case, answer style, key concepts, excluded topics) via Claude Sonnet or Gemini 2.5 Pro — one premium LLM call.
4. **Generates strategic query archetypes** (5 categories × N variants per category) grounded in the wiki sample + thesis.
5. **Pre-computes answer chunks**: for each archetype the LLM teacher synthesizes an "ideal answer" citing the evidence wiki slugs inline as `[[slug]]`.
6. **Embeds** pages + chunks with E5 multilingual (off-the-shelf, runs in-actor).
7. **Optionally deduplicates** via cross-tenant Pinecone RAG — chunks similar to ones already learned from other wikis are skipped (network-effect learning).
8. **Exports** a portable `compiled_wiki.zip` plus an `mcp_manifest.json` you paste into your AI agent's config.

## Incremental updates (v0.3)

Re-compile only what changed. Pass the previous `compiled_wiki.zip` back via
`priorBundleUrl` and the compiler diffs your wiki against it, re-embeds **only
added/changed pages**, and regenerates **only the answer chunks whose evidence
moved** — typically far cheaper than a full recompile on small edits (the
premium thesis call and unchanged embeddings are reused). The bundle's
`state.json` is the diff baseline, like a lockfile; if the embedding model
changed or too much of the wiki changed, it falls back to a full rebuild
automatically. More on the approach at **[42rows.com](https://42rows.com)**.

## Per-tenant retrieval store (v0.3, optional)

Set `tenantId` (and optionally `collection`, default `company-wiki`) and the
compiled vectors are synced to a [Pinecone](https://www.pinecone.io) index
(= the `collection`) under a namespace (= the `tenantId`) — Pinecone's native
multi-tenant isolation. Your agent then retrieves from `(collection, tenant)`
at query time. Mapping: **collection → index, tenant → namespace**. Leave
`tenantId` empty to just produce the portable bundle (no Pinecone write).

## Vertex AI

Gemini runs via AI Studio (`googleApiKey`) by default, or through **Vertex AI**
when `GOOGLE_GENAI_USE_VERTEXAI=true` (with `GOOGLE_CLOUD_PROJECT` +
service-account credentials) — required where the AI Studio API is
geo-restricted.

## Why this exists

- **Karpathy LLM Wiki** is brilliant but slow (~2s per query) and locked to one LLM session.
- **Generic vector RAG** treats text as opaque chunks — no purpose, no provenance, no structure.
- **AI agents** (Claude Desktop, Cursor, Cline) need *persistent* knowledge that survives session restarts.

The Wiki Compiler is the offline build step. The runtime is [`mariowiki-mcp`](https://github.com/42ROWS/wiki-embedded) — an MIT-licensed pip package that reads `compiled_wiki.zip` and serves it over Model Context Protocol.

## Quick start

### 1. Run the actor

Pick a source, paste a 1-sentence intent, hit run.

```json
{
  "wikiSource": "github",
  "githubRepo": "your-org/your-knowledge-wiki",
  "githubBranch": "main",
  "thesisIntent": "Help a B2B sales rep find prospects and explain why they're hot.",
  "answerStyle": "concise_actionable",
  "chunkArchetypes": ["ranking_top_n", "comparative", "explanatory", "tactical_recipe"],
  "chunksPerArchetype": 4,
  "llmProvider": "google",
  "googleApiKey": "<your Gemini key>"
}
```

The run produces:

- `compiled_wiki.zip` in the Key-Value store (pages + chunks + embeddings + manifest)
- `mcp_manifest.json` with a plug-and-play config snippet

### 2. Plug into your AI agent

Drop the snippet into `claude_desktop_config.json` (or `~/.cursor/mcp.json` / `cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "mariowiki": {
      "command": "mariowiki-mcp",
      "args": ["--compiled-wiki", "<URL_FROM_OUTPUT>"]
    }
  }
}
```

Restart. Ask your agent: *"use the mariowiki tool to find the top prospects."* It calls `query_wiki_answer(...)` and gets a pre-computed, source-cited chunk back in ~50ms.

## Pricing

| Mode | LLM cost | Apify cost |
|---|---|---|
| BYOK Google Gemini | ~$0.05–$0.20 (paid to Google) | per the actor's pricing tier |
| BYOK Anthropic Claude | ~$0.20–$0.60 (paid to Anthropic) | per the actor's pricing tier |
| Hosted | included | flat $5 / compile (uses our Gemini quota) |

E5 embeddings run in the actor (free, off-the-shelf). Pinecone cross-tenant dedup is optional and free on Pinecone's serverless free tier.

## Architecture

```
Markdown wiki
      ↓ parse + crossref
Wiki pages (atomic, normalized)
      ↓ LLM teacher offline (one premium thesis call + N cheap chunk calls)
Chunks (materialized views — pre-computed, thesis-conditional, citing slugs)
      ↓ E5 embedder
Compiled wiki bundle (pages + chunks + embeddings + manifest)
      ↓ Apify KV store
mariowiki-mcp (consumer, pip package) → Claude Desktop / Cursor / Cline
```

### Database analogy

| DB concept | Wiki Compiler |
|---|---|
| Normalized tables | Wiki markdown pages |
| Materialized view | Pre-computed answer chunk |
| Cache layer | E5 embeddings of chunks |
| CQRS write-side | Editable markdown sources |
| CQRS read-side | Compiled bundle served over MCP |
| B-tree index | E5 cosine similarity |

## Prior art (honest)

This compiler combines well-known patterns into a single delivery:

- **Karpathy LLM Wiki** (2024 gist) — the wiki-as-knowledge pattern
- **INSTRUCTOR** (Su et al., NAACL 2023) — instruction-conditioned embeddings
- **HyDE** (Gao et al., 2022) — hypothetical document embeddings
- **GraphRAG** (Microsoft, 2024) — pre-computed community summaries
- **RAPTOR** (Sarthi et al., ICLR 2024) — hierarchical summary chunks
- **MCP** (Anthropic, 2024) — tool delivery standard

Novel contribution: **persistent, thesis-conditional, pre-computed answer chunks** packaged as a portable bundle with a learning cross-tenant RAG dedup layer.

## License

MIT. See `LICENSE`.

## Author

Mario Brosco — [42rows S.r.l.](https://42rows.com) · [info@42rows.com](mailto:info@42rows.com)
