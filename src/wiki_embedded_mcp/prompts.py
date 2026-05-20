"""MCP prompt templates exposed by Wiki Embedded MCP.

A Prompt is a parameterized message template the client (Claude Desktop / Cursor /
Cline) can surface to the user as a quick action — e.g. a "Summarize this wiki"
button that the user clicks instead of typing a free-form query.

Templates are intentionally short and call other tools for evidence so the
runtime stays fast and grounded.
"""
from __future__ import annotations

from typing import Any

from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    TextContent,
)

# ── Prompt declarations ────────────────────────────────────────────────────
PROMPTS: list[Prompt] = [
    Prompt(
        name="summarize_wiki",
        description=(
            "Produce a concise overview of the wiki: its thesis, key concepts, "
            "and the top categories of pages. The assistant should call "
            "`get_thesis` and `list_wiki_pages` to ground the summary."
        ),
        arguments=[],
    ),
    Prompt(
        name="ask_about",
        description=(
            "Ask a focused question and have the assistant cite the wiki. "
            "Uses `query_wiki_answer` for a pre-computed chunk if available, "
            "and falls back to `query_wiki_pages` for raw evidence."
        ),
        arguments=[
            PromptArgument(
                name="question",
                description="The natural-language question to answer using the wiki.",
                required=True,
            )
        ],
    ),
    Prompt(
        name="compare_topics",
        description=(
            "Compare two topics or entities in the wiki. The assistant should "
            "retrieve evidence for both via `query_wiki_pages` and present a "
            "structured side-by-side comparison."
        ),
        arguments=[
            PromptArgument(name="topic_a", description="First topic or slug.", required=True),
            PromptArgument(name="topic_b", description="Second topic or slug.", required=True),
        ],
    ),
]


def render_prompt(name: str, arguments: dict[str, Any] | None) -> GetPromptResult:
    """Materialize a prompt template with the supplied arguments."""
    args = arguments or {}

    if name == "summarize_wiki":
        text = (
            "Please summarize this wiki for me.\n\n"
            "Steps:\n"
            "1. Call `get_thesis` to learn the wiki's purpose and audience.\n"
            "2. Call `list_wiki_pages` (no kind filter) to see what's available.\n"
            "3. Optionally call `list_wiki_pages(kind=\"chunk\")` to see pre-computed answers.\n"
            "4. Produce a 5-bullet summary covering: purpose, target audience, key "
            "concepts, scope/excluded topics, notable answer chunks."
        )
    elif name == "ask_about":
        question = str(args.get("question", "")).strip()
        if not question:
            raise ValueError("ask_about requires a non-empty 'question' argument")
        text = (
            f"Answer this question using the wiki: {question}\n\n"
            "Steps:\n"
            "1. Call `get_thesis` once to calibrate tone (only if not already done this session).\n"
            "2. Call `query_wiki_answer` with the question; if a chunk is returned, base your "
            "answer on it and cite the slugs it mentions.\n"
            "3. If no answer chunk, call `query_wiki_pages` and synthesize from the top results.\n"
            "4. Always cite at least 2 wiki slugs in the form [[slug]]."
        )
    elif name == "compare_topics":
        topic_a = str(args.get("topic_a", "")).strip()
        topic_b = str(args.get("topic_b", "")).strip()
        if not topic_a or not topic_b:
            raise ValueError("compare_topics requires both 'topic_a' and 'topic_b'")
        text = (
            f"Compare these two topics using the wiki: {topic_a} vs {topic_b}.\n\n"
            "Steps:\n"
            "1. Call `query_wiki_pages` for each topic, top_k=3.\n"
            "2. If either topic looks like a slug already, call `read_wiki_page` directly.\n"
            "3. Produce a structured comparison: similarities, differences, when to "
            "prefer one over the other. Cite slugs inline as [[slug]]."
        )
    else:
        raise ValueError(f"unknown prompt: {name}")

    return GetPromptResult(
        description=next((p.description for p in PROMPTS if p.name == name), None),
        messages=[
            PromptMessage(role="user", content=TextContent(type="text", text=text)),
        ],
    )
