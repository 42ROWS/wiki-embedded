"""LLM module - multi-provider support for code generation."""
from polars_runner.llm.client import (
    LLMService,
    create_client,
    GroqClient,
    AnthropicClient,
    OpenAIClient,
    GoogleClient,
)
from polars_runner.llm.providers import (
    GroqProvider,
    AnthropicProvider,
    OpenAIProvider,
    GoogleProvider,
    create_provider,
)
from polars_runner.llm.prompts import (
    POLARS_SYSTEM_PROMPT,
    build_user_prompt,
    build_error_recovery_prompt,
    build_few_shot_section,
)
from polars_runner.llm.base import BaseLLMProvider

__all__ = [
    # Client (sync)
    "LLMService",
    "create_client",
    "GroqClient",
    "AnthropicClient",
    "OpenAIClient",
    "GoogleClient",
    # Providers (async)
    "BaseLLMProvider",
    "GroqProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "create_provider",
    # Prompts
    "POLARS_SYSTEM_PROMPT",
    "build_user_prompt",
    "build_error_recovery_prompt",
    "build_few_shot_section",
]
