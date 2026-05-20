"""Derive a structured thesis from sample pages + user intent.

The thesis is the "purpose lens" of the wiki: who reads it, what they're trying
to answer, in what style, with what excluded. It's frontloaded at setup time
(LLM-premium one-shot) and then influences chunk generation + retrieval ranking.

Output schema (returned dict + saved to thesis.md frontmatter):
    target_audience: str
    primary_use_case: str
    answer_style: str
    key_concepts: list[str]
    excluded_topics: list[str]
    summary: str  (one-line)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

THESIS_PROMPT = """You are a wiki strategist. Read the user intent and a stratified sample of pages from the wiki, then derive a structured thesis.

USER INTENT:
{intent}

PREFERRED ANSWER STYLE:
{answer_style}

WIKI SAMPLE (titles + excerpts):
{sample}

Task: produce a structured thesis as JSON. Strict schema:
{{
  "target_audience": "1 sentence — who will query this wiki",
  "primary_use_case": "1 sentence — what they're trying to answer",
  "answer_style": "1 sentence — tone, depth, citations expected",
  "key_concepts": ["5-15 core terms / entities specific to this wiki"],
  "excluded_topics": ["3-7 topics this wiki should NOT answer (anti-scope)"],
  "summary": "1 sentence — the whole thesis distilled"
}}

Rules:
- Anchor everything in the actual sample content (no hallucination of topics not present)
- Be domain-specific to THIS wiki — adapt to whatever domain you find in the sample
  (it could be code docs, sales playbook, research notes, recipes, anything)
- Use neutral vocabulary: do not assume a domain (no "prospects" if it's a recipe wiki,
  no "ingredients" if it's a codebase). Pick the vocabulary that fits the sample
- excluded_topics should be meaningful boundaries (not "irrelevant stuff")

Output ONLY the JSON object. No preamble, no code fence, no extra text."""


@dataclass
class Thesis:
    target_audience: str = ""
    primary_use_case: str = ""
    answer_style: str = ""
    key_concepts: list[str] = field(default_factory=list)
    excluded_topics: list[str] = field(default_factory=list)
    summary: str = ""
    model: str = ""
    intent_raw: str = ""

    @property
    def hash(self) -> str:
        """Stable SHA256 of the semantic content (excludes model field)."""
        payload = json.dumps(
            {
                "target_audience": self.target_audience,
                "primary_use_case": self.primary_use_case,
                "answer_style": self.answer_style,
                "key_concepts": sorted(self.key_concepts),
                "excluded_topics": sorted(self.excluded_topics),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Serialize as a thesis.md file (frontmatter + body)."""
        import frontmatter
        post = frontmatter.Post(
            self.summary,
            kind="thesis",
            thesis_hash=self.hash,
            target_audience=self.target_audience,
            primary_use_case=self.primary_use_case,
            answer_style=self.answer_style,
            key_concepts=self.key_concepts,
            excluded_topics=self.excluded_topics,
            model=self.model,
            intent_raw=self.intent_raw,
        )
        return frontmatter.dumps(post)


def _format_sample(pages: list[dict[str, Any]], max_chars_per_page: int = 400) -> str:
    parts: list[str] = []
    for p in pages:
        title = (p.get("title") or "").strip()
        slug = p.get("slug", "")
        body = (p.get("body") or "")[:max_chars_per_page].strip()
        parts.append(f"### [{slug}] {title}\n{body}")
    return "\n\n".join(parts)


def derive_thesis_google(
    intent: str,
    answer_style: str,
    sample_pages: list[dict[str, Any]],
    api_key: str,
    model: str = "gemini-2.5-pro",
) -> Thesis:
    """Derive thesis via Google Gemini (one-shot, premium quality)."""
    from google.genai import types as gtypes

    from wiki_compiler.wiki._google import genai_client
    client = genai_client(api_key)
    prompt = THESIS_PROMPT.format(
        intent=intent,
        answer_style=answer_style,
        sample=_format_sample(sample_pages),
    )
    cfg = gtypes.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2000,
        response_mime_type="application/json",
    )
    r = client.models.generate_content(model=model, contents=prompt, config=cfg)
    parsed = json.loads(r.text or "{}")
    return Thesis(
        target_audience=str(parsed.get("target_audience", "")),
        primary_use_case=str(parsed.get("primary_use_case", "")),
        answer_style=str(parsed.get("answer_style", "")),
        key_concepts=[str(x) for x in parsed.get("key_concepts", [])][:20],
        excluded_topics=[str(x) for x in parsed.get("excluded_topics", [])][:10],
        summary=str(parsed.get("summary", "")),
        model=model,
        intent_raw=intent,
    )


def derive_thesis_anthropic(
    intent: str,
    answer_style: str,
    sample_pages: list[dict[str, Any]],
    api_key: str,
    model: str = "claude-sonnet-4-6",
) -> Thesis:
    """Derive thesis via Anthropic Claude (one-shot, premium quality)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = THESIS_PROMPT.format(
        intent=intent,
        answer_style=answer_style,
        sample=_format_sample(sample_pages),
    )
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in msg.content if block.type == "text")
    parsed = json.loads(text)
    return Thesis(
        target_audience=str(parsed.get("target_audience", "")),
        primary_use_case=str(parsed.get("primary_use_case", "")),
        answer_style=str(parsed.get("answer_style", "")),
        key_concepts=[str(x) for x in parsed.get("key_concepts", [])][:20],
        excluded_topics=[str(x) for x in parsed.get("excluded_topics", [])][:10],
        summary=str(parsed.get("summary", "")),
        model=model,
        intent_raw=intent,
    )


def derive_thesis(
    provider: Literal["google", "anthropic"],
    intent: str,
    answer_style: str,
    sample_pages: list[dict[str, Any]],
    api_key: str,
) -> Thesis:
    """Dispatch to the configured LLM provider."""
    if provider == "google":
        return derive_thesis_google(intent, answer_style, sample_pages, api_key)
    if provider == "anthropic":
        return derive_thesis_anthropic(intent, answer_style, sample_pages, api_key)
    raise ValueError(f"Unsupported thesis provider: {provider}")
