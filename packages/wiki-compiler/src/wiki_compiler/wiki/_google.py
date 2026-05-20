"""Gemini client factory — Vertex AI or AI Studio (api_key), env-driven.

Set ``GOOGLE_GENAI_USE_VERTEXAI=true`` (plus ``GOOGLE_CLOUD_PROJECT``,
``GOOGLE_CLOUD_LOCATION`` and service-account creds via
``GOOGLE_APPLICATION_CREDENTIALS`` / ADC) to route Gemini calls through Vertex AI
— required where the AI Studio API (``generativelanguage``) is geo-restricted.
Otherwise the call uses the provided ``api_key`` (AI Studio / BYOK).
"""
from __future__ import annotations

import os


def use_vertex() -> bool:
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in ("true", "1", "yes")


def genai_client(api_key: str | None = None):
    """Return a configured ``google.genai`` client (Vertex AI or AI Studio)."""
    from google import genai

    if use_vertex():
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    return genai.Client(api_key=api_key)
