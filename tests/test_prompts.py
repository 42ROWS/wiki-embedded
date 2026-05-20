"""Tests for prompts.py — MCP Prompt rendering."""
from __future__ import annotations

import pytest

from wiki_embedded_mcp.prompts import PROMPTS, render_prompt


def test_prompts_have_expected_names() -> None:
    names = {p.name for p in PROMPTS}
    assert names == {"summarize_wiki", "ask_about", "compare_topics"}


def test_render_summarize_wiki_no_args() -> None:
    result = render_prompt("summarize_wiki", arguments=None)
    assert result.messages
    assert "summarize" in result.messages[0].content.text.lower()


def test_render_ask_about_with_question() -> None:
    result = render_prompt("ask_about", arguments={"question": "What is alpha?"})
    text = result.messages[0].content.text
    assert "What is alpha?" in text
    assert "query_wiki_answer" in text


def test_render_ask_about_missing_arg_raises() -> None:
    with pytest.raises(ValueError):
        render_prompt("ask_about", arguments={})


def test_render_compare_topics_requires_both() -> None:
    with pytest.raises(ValueError):
        render_prompt("compare_topics", arguments={"topic_a": "x"})


def test_render_compare_topics_ok() -> None:
    result = render_prompt(
        "compare_topics", arguments={"topic_a": "alpha", "topic_b": "beta"}
    )
    text = result.messages[0].content.text
    assert "alpha" in text and "beta" in text


def test_render_unknown_prompt_raises() -> None:
    with pytest.raises(ValueError):
        render_prompt("does_not_exist", arguments=None)
