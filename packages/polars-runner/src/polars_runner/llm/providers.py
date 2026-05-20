"""
LLM Provider implementations.
Each provider implements the BaseLLMProvider interface.
"""
from __future__ import annotations

from typing import Any

from ..core import LLMProvider, LLMResponse
from .base import BaseLLMProvider


# =============================================================================
# GROQ PROVIDER (Free tier available)
# =============================================================================

class GroqProvider(BaseLLMProvider):
    """
    Groq provider using Llama 3.3 70B.
    Free tier: 14,400 requests/day, 500K tokens/day.
    """
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GROQ
    
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call Groq API."""
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=self._api_key)
        
        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            code=choice.message.content or "",
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model=self._model,
        )


# =============================================================================
# ANTHROPIC PROVIDER (Claude)
# =============================================================================

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.
    Best for code generation (72.7% SWE-bench).
    """
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC
    
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call Anthropic API."""
        from anthropic import AsyncAnthropic
        
        client = AsyncAnthropic(api_key=self._api_key)
        
        response = await client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        
        # Extract text from response
        content = response.content[0]
        code = content.text if hasattr(content, "text") else str(content)
        
        return LLMResponse(
            code=code,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model=self._model,
        )


# =============================================================================
# OPENAI PROVIDER (GPT-4o)
# =============================================================================

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI GPT-4o provider.
    Largest ecosystem, reliable performance.
    """
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI
    
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call OpenAI API."""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self._api_key)
        
        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        choice = response.choices[0]
        usage = response.usage
        
        return LLMResponse(
            code=choice.message.content or "",
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model=self._model,
        )


# =============================================================================
# GOOGLE PROVIDER (Gemini)
# =============================================================================

class GoogleProvider(BaseLLMProvider):
    """
    Google Gemini provider.
    Fast, 1M context window.
    """
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GOOGLE
    
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Call Google Gemini API."""
        import google.generativeai as genai
        
        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            self._model,
            system_instruction=system_prompt,
        )
        
        # Gemini doesn't have native async, use sync in async context
        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=self._max_tokens,
            ),
        )
        
        # Extract token counts
        usage = response.usage_metadata
        
        return LLMResponse(
            code=response.text,
            tokens_input=usage.prompt_token_count if usage else 0,
            tokens_output=usage.candidates_token_count if usage else 0,
            model=self._model,
        )


# =============================================================================
# PROVIDER FACTORY
# =============================================================================

_PROVIDER_CLASSES: dict[LLMProvider, type[BaseLLMProvider]] = {
    LLMProvider.GROQ: GroqProvider,
    LLMProvider.ANTHROPIC: AnthropicProvider,
    LLMProvider.OPENAI: OpenAIProvider,
    LLMProvider.GOOGLE: GoogleProvider,
}


def create_provider(
    provider: LLMProvider,
    api_key: str | None,
    model: str | None = None,
) -> BaseLLMProvider:
    """
    Factory function to create LLM provider.
    
    Args:
        provider: Provider type
        api_key: API key
        model: Optional model override
        
    Returns:
        Configured provider instance
    """
    provider_class = _PROVIDER_CLASSES.get(provider)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}")
    
    return provider_class(api_key=api_key, model=model)
