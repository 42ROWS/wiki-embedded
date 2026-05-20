# Quickstart — Wiki Embedded MCP

3-minute setup to query your wiki from Claude Desktop / Cursor / Cline.

## 1. Install

```bash
pip install 'wiki-embedded-mcp[cloud]'
```

Or with Docker (no Python on the host):

```bash
docker pull ghcr.io/42rows/wiki-embedded-mcp:latest
```

## 2. Pick a wiki source

**A — Compiled bundle from the [42rows Wiki Compiler](https://42rows.com/wiki)** (instant boot):

```bash
export PINECONE_API_KEY="..."
wiki-embedded-mcp --compiled-wiki https://api.apify.com/.../compiled_wiki.zip
```

**B — Local markdown directory** (parsed + embedded at startup):

```bash
wiki-embedded-mcp --wiki ./my-knowledge-base
```

## 3. Wire it into your MCP client

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "wiki-embedded": {
      "command": "wiki-embedded-mcp",
      "args": ["--compiled-wiki", "/absolute/path/to/compiled_wiki.zip"],
      "env": { "PINECONE_API_KEY": "..." }
    }
  }
}
```

Cursor: `~/.cursor/mcp.json`. Cline: workspace settings. Restart the client.

## 4. Try it

In Claude Desktop, ask:

> "Use the wiki-embedded MCP to summarize the wiki."

The agent calls `summarize_wiki` (a built-in Prompt), which under the hood calls `get_thesis` + `list_wiki_pages` and produces a grounded summary with citations.

## Troubleshooting

**`PINECONE_API_KEY env var required`** → the compiled bundle uses `pinecone:` embeddings. Get a free key at [pinecone.io](https://www.pinecone.io/) (free tier covers ~5M tokens/month).

**Embedding download is slow on first run (local mode)** → first invocation downloads `intfloat/multilingual-e5-base` (~500 MB) from HuggingFace. Cached after that.

**Slow queries on wikis >50K pages** → install the FAISS extra: `pip install 'wiki-embedded-mcp[faiss]'`.

## Next steps

- [README](./README.md) — full architecture, benchmarks, prior art
- [Hosted at 42rows.com/wiki](https://42rows.com/wiki) — compile without managing infra
- [42rows Wiki Compiler on Apify Store](https://apify.com/salesmart-srl/42rows-wiki-compiler) — the offline compiler
