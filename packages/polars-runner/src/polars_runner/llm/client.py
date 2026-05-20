"""
LLM client abstraction with multi-provider support.
BYOK (Bring Your Own Key) + Hosted mode architecture.
"""
import time
import re
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from polars_runner.core.constants import (
    LLMProvider,
    LLM_CONFIGS,
    LIMITS,
)
from polars_runner.core.exceptions import LLMGenerationError
from polars_runner.core.models import GenerationResult, SchemaInfo
from polars_runner.llm.prompts import (
    POLARS_SYSTEM_PROMPT,
    build_user_prompt,
    build_error_recovery_prompt,
    build_error_recovery_prompt_v2,
)
from polars_runner.core.error_analyzer import ErrorAnalysis


# =============================================================================
# PROTOCOL / INTERFACE
# =============================================================================

@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int]:
        """
        Generate completion.
        
        Returns:
            Tuple of (response_text, tokens_used)
        """
        ...


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================

class GroqClient:
    """Groq API client (OpenAI-compatible)."""
    
    def __init__(self, api_key: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._config = LLM_CONFIGS[LLMProvider.GROQ]
    
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        response = self._client.chat.completions.create(
            model=LLMProvider.GROQ.model_id,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens


class AnthropicClient:
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)
        self._config = LLM_CONFIGS[LLMProvider.ANTHROPIC]
    
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        response = self._client.messages.create(
            model=LLMProvider.ANTHROPIC.model_id,
            max_tokens=self._config.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens


class OpenAIClient:
    """OpenAI API client."""
    
    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._config = LLM_CONFIGS[LLMProvider.OPENAI]
    
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        response = self._client.chat.completions.create(
            model=LLMProvider.OPENAI.model_id,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens


class GoogleClient:
    """Google Gemini API client using new unified SDK."""

    def __init__(self, api_key: str, model_id: str | None = None):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id or LLMProvider.GOOGLE.model_id
        self._config = LLM_CONFIGS[LLMProvider.GOOGLE]

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        from google.genai import types

        # Combine system + user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
            )
        )

        text = response.text

        # Get token count from usage metadata
        tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = getattr(response.usage_metadata, 'total_token_count', 0)
        if not tokens:
            # Fallback estimate (~4 chars per token)
            tokens = (len(full_prompt) + len(text)) // 4

        return text, tokens


class GoogleFlashLiteClient:
    """
    Google Gemini 2.5 Flash-Lite client for hosted mode (Basic tier).

    This is our internal hosted LLM - users don't need to provide API key.
    Cost: $0.10/1M input, $0.40/1M output tokens
    Rate limit: 30M TPM (Tier 3)
    """

    def __init__(self, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_id = LLMProvider.GOOGLE_FLASH_LITE.model_id
        self._config = LLM_CONFIGS[LLMProvider.GOOGLE_FLASH_LITE]

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        from google.genai import types

        # Combine system + user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
            )
        )

        text = response.text

        # Get token count from usage metadata
        tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = getattr(response.usage_metadata, 'total_token_count', 0)
        if not tokens:
            # Fallback estimate (~4 chars per token)
            tokens = (len(full_prompt) + len(text)) // 4

        return text, tokens


class GoogleProClient:
    """
    Google Gemini 2.5 Pro client with reasoning for Premium tier.

    Features:
    - Extended Thinking (reasoning with thinking_budget)
    - Google Search grounding (always enabled)
    - 2M token context window (vs 128K for Flash-Lite)
    - Better for complex transformations

    Cost: $1.25/1M input, $5.00/1M output tokens
    Grounding: $35 per 1,000 queries (billed per search query)
    Rate limit: 30M TPM (Tier 3)
    """

    def __init__(self, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_id = LLMProvider.GOOGLE_PRO.model_id
        self._config = LLM_CONFIGS[LLMProvider.GOOGLE_PRO]

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        """Generate with extended thinking and Google Search grounding."""
        from google.genai import types

        # Combine system + user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # ✅ Generate with Thinking + Google Search grounding
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
                # ✅ Enable extended thinking (reasoning)
                thinking_config=types.ThinkingConfig(
                    thinking_budget=1024  # Token budget for reasoning (128-32768)
                ),
                # ✅ Enable Google Search grounding
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        text = response.text

        # Log grounding metadata if available
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata

                    # Log web search queries executed
                    if hasattr(metadata, 'web_search_queries') and metadata.web_search_queries:
                        num_queries = len(metadata.web_search_queries)
                        print(f"🔍 Google Search: {num_queries} queries executed")
                        for i, query in enumerate(metadata.web_search_queries, 1):
                            print(f"   {i}. {query}")

                    # Log grounding chunks (sources)
                    if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                        num_sources = len(metadata.grounding_chunks)
                        print(f"📚 Sources: {num_sources} web pages used")
        except Exception as e:
            # Don't fail on grounding metadata logging
            print(f"⚠️ Could not parse grounding metadata: {e}")

        # Get token count from usage metadata
        tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens = getattr(response.usage_metadata, 'total_token_count', 0)
        if not tokens:
            # Fallback estimate (~4 chars per token)
            tokens = (len(full_prompt) + len(text)) // 4

        return text, tokens


# =============================================================================
# CLIENT FACTORY
# =============================================================================

def create_client(provider: LLMProvider, api_key: str) -> LLMClientProtocol:
    """Factory for creating LLM clients."""
    match provider:
        case LLMProvider.GROQ:
            return GroqClient(api_key)
        case LLMProvider.ANTHROPIC:
            return AnthropicClient(api_key)
        case LLMProvider.OPENAI:
            return OpenAIClient(api_key)
        case LLMProvider.GOOGLE:
            return GoogleClient(api_key)
        case LLMProvider.GOOGLE_FLASH_LITE:
            return GoogleFlashLiteClient(api_key)
        case LLMProvider.GOOGLE_PRO:
            return GoogleProClient(api_key)
        case _:
            raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# MAIN LLM SERVICE
# =============================================================================

class LLMService:
    """
    High-level LLM service for code generation.
    Handles retries, fallback, and code extraction.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        fallback_provider: LLMProvider | None = None,
        fallback_api_key: str | None = None,
    ):
        self._primary_provider = provider
        self._primary_client = create_client(provider, api_key)
        
        self._fallback_provider = fallback_provider
        self._fallback_client = None
        if fallback_provider and fallback_api_key:
            self._fallback_client = create_client(fallback_provider, fallback_api_key)
    
    @property
    def provider(self) -> LLMProvider:
        """Get the primary provider."""
        return self._primary_provider
    
    @property
    def is_hosted(self) -> bool:
        """Check if using hosted (non-BYOK) provider."""
        return self._primary_provider.is_hosted
    
    def generate_polars_code(
        self,
        user_prompt: str,
        schema: SchemaInfo,
        similar_codes: list | None = None,  # RAG context from Pinecone
        max_retries: int = LIMITS.max_retries,
        multi_table_schemas: dict[str, SchemaInfo] | None = None,  # For multi-table JOIN
        previous_code: str | None = None,  # For execution error recovery
        previous_error: str | None = None,  # For execution error recovery
        error_analysis: ErrorAnalysis | None = None,  # Structured error analysis
    ) -> GenerationResult:
        """
        Generate Polars transformation code.

        Args:
            user_prompt: User's transformation request
            schema: Dataset schema information (used for single-table mode)
            similar_codes: List of similar transformations from RAG (optional)
            max_retries: Max generation attempts
            multi_table_schemas: Dict of {table_name: SchemaInfo} for multi-table JOIN
            previous_code: Previous code that failed execution (for error recovery)
            previous_error: Error message from failed execution (for error recovery)
            error_analysis: Structured error analysis from ErrorAnalyzer (for intelligent recovery)

        Returns:
            GenerationResult with code and metadata
        """
        start_time = time.perf_counter()

        # Prepare multi-table schema descriptions if provided
        multi_table_desc = None
        if multi_table_schemas:
            multi_table_desc = {
                name: info.to_llm_description()
                for name, info in multi_table_schemas.items()
            }

        # Check if this is an error recovery call
        if previous_code and previous_error:
            # Use intelligent error recovery prompt (V2) with structured analysis
            prompt = build_error_recovery_prompt_v2(
                user_prompt=user_prompt,
                schema_description=schema.to_llm_description(),
                previous_code=previous_code,
                error_message=previous_error,
                error_analysis=error_analysis,  # Structured error context
                multi_table_schemas=multi_table_desc,  # Now passed to recovery!
            )
        else:
            # Build initial prompt (with RAG context if available)
            prompt = build_user_prompt(
                user_prompt=user_prompt,
                schema_description=schema.to_llm_description(),
                row_count=schema.row_count,
                similar_codes=similar_codes,  # Pass RAG context
                include_examples=True,
                multi_table_schemas=multi_table_desc,
            )

        last_error: str | None = None
        last_code: str | None = None
        total_tokens = 0
        attempts = 0
        provider_used = self._primary_provider

        # Try primary provider
        for attempt in range(max_retries):
            attempts += 1

            try:
                if last_error and last_code:
                    # Use error recovery prompt for generation validation errors
                    # Note: This handles code validation errors, not execution errors
                    prompt = build_error_recovery_prompt_v2(
                        user_prompt=user_prompt,
                        schema_description=schema.to_llm_description(),
                        previous_code=last_code,
                        error_message=last_error,
                        error_analysis=None,  # No structured analysis for validation errors
                        multi_table_schemas=multi_table_desc,  # Pass multi-table context
                    )

                response, tokens = self._primary_client.generate(
                    system_prompt=POLARS_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                total_tokens += tokens

                # Extract code from response
                code = self._extract_code(response)

                # Basic validation
                self._validate_code_structure(code)

                return GenerationResult(
                    code=code,
                    provider_used=provider_used,
                    tokens_used=total_tokens,
                    generation_time_ms=int((time.perf_counter() - start_time) * 1000),
                    attempts=attempts,
                )

            except Exception as e:
                last_error = str(e)
                last_code = code if 'code' in dir() else None
        
        # Try fallback if available
        if self._fallback_client:
            provider_used = self._fallback_provider
            
            try:
                prompt = build_user_prompt(
                    user_prompt=user_prompt,
                    schema_description=schema.to_llm_description(),
                    row_count=schema.row_count,
                    similar_codes=similar_codes,  # Pass RAG context to fallback too
                    include_examples=True,
                )
                
                response, tokens = self._fallback_client.generate(
                    system_prompt=POLARS_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                total_tokens += tokens
                attempts += 1
                
                code = self._extract_code(response)
                self._validate_code_structure(code)
                
                return GenerationResult(
                    code=code,
                    provider_used=provider_used,
                    tokens_used=total_tokens,
                    generation_time_ms=int((time.perf_counter() - start_time) * 1000),
                    attempts=attempts,
                )
                
            except Exception as e:
                last_error = str(e)
        
        # All attempts failed
        raise LLMGenerationError(
            message=f"Code generation failed after {attempts} attempts: {last_error}",
            provider=provider_used.value,
            prompt_preview=user_prompt[:200],
            attempts=attempts,
        )
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        response = response.strip()
        
        # Try to extract from markdown code block
        if "```python" in response:
            match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        if "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Return as-is if no code blocks
        return response
    
    def _validate_code_structure(self, code: str) -> None:
        """Basic validation of code structure."""
        if not code:
            raise ValueError("Empty code generated")
        
        if "result" not in code:
            raise ValueError("Code must assign to 'result' variable")
        
        # Check for obvious Pandas patterns
        pandas_patterns = [
            r"\.groupby\(",  # Should be group_by
            r"df\[['\"]",    # df["col"] style
            r"\.iloc\[",     # iloc indexing
            r"\.loc\[",      # loc indexing
        ]
        
        for pattern in pandas_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Pandas syntax detected: {pattern}")
