"""
Abstract base class for LLM providers.
Defines the interface all providers must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

from polars_runner.core.constants import (
    DEFAULT_MODELS,
    LLMModel,
    LLMProvider,
    ProcessingLimits,
)
from polars_runner.core.types import LLMResponse
from .prompts import (
    POLARS_SYSTEM_PROMPT,
    build_user_prompt,
    build_error_recovery_prompt,
)


# Singleton limits
_LIMITS: Final[ProcessingLimits] = ProcessingLimits()


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM provider implementations.
    
    All providers must implement the _call_api method.
    Common logic for prompt building and response handling is shared.
    """
    
    def __init__(
        self,
        api_key: str | None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """
        Initialize provider.
        
        Args:
            api_key: API key for the provider
            model: Model name (uses default if None)
            max_tokens: Maximum tokens in response
        """
        self._api_key = api_key
        self._model = model or self._default_model
        self._max_tokens = max_tokens or _LIMITS.MAX_TOKENS_OUTPUT
        
        # Validate API key if required
        if self._requires_api_key and not api_key:
            raise ValueError(f"API key required for {self.provider.value}")
    
    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        ...
    
    @property
    def _default_model(self) -> str:
        """Get default model for this provider."""
        return DEFAULT_MODELS[self.provider].value
    
    @property
    def _requires_api_key(self) -> bool:
        """Whether this provider requires an API key."""
        # Groq has a free tier
        return self.provider != LLMProvider.GROQ
    
    @abstractmethod
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Call the provider's API.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            
        Returns:
            LLMResponse with generated code and token counts
        """
        ...
    
    async def generate_code(
        self,
        user_prompt: str,
        schema_info: str,
        row_count: int = 0,
        previous_error: str | None = None,
    ) -> LLMResponse:
        """
        Generate Polars transformation code.
        
        Args:
            user_prompt: User's transformation request
            schema_info: Formatted schema string
            row_count: Number of rows in dataset
            previous_error: Error from previous attempt (for retry)
            
        Returns:
            LLMResponse with generated code
        """
        system_prompt = POLARS_SYSTEM_PROMPT
        
        if previous_error:
            prompt = build_error_recovery_prompt(
                user_prompt=user_prompt,
                schema_description=schema_info,
                previous_code="",  # Not tracked in this version
                error_message=previous_error,
            )
        else:
            prompt = build_user_prompt(
                user_prompt=user_prompt,
                schema_description=schema_info,
                row_count=row_count,
            )
        
        return await self._call_api(system_prompt, prompt)
    
    @property
    def model_name(self) -> str:
        """Get current model name."""
        return self._model
