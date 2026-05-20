"""
Configuration management with environment variable support.
Single source of truth for all configuration values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import (
    API_KEY_ENV_VARS,
    API_KEY_INPUT_FIELDS,
    LLMProvider,
    OutputFormat,
    ProcessingLimits,
)


# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def is_apify_environment() -> bool:
    """Check if running on Apify platform."""
    return os.getenv("APIFY_IS_AT_HOME", "").lower() == "true"


def get_environment() -> str:
    """Get current environment name."""
    return os.getenv("ENVIRONMENT", "development")


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass(frozen=True, slots=True)
class Config:
    """
    Application configuration.
    Immutable after creation for thread safety.
    """
    
    # Environment
    environment: str
    is_apify: bool
    log_level: str
    
    # API Keys
    groq_api_key: str | None
    anthropic_api_key: str | None
    openai_api_key: str | None
    google_api_key: str | None
    
    # Processing
    max_retries: int
    code_timeout_seconds: int
    max_memory_mb: int
    streaming_threshold_bytes: int
    streaming_chunk_size: int
    
    # Output
    output_dir: Path
    
    @classmethod
    def from_environment(cls) -> Config:
        """Create config from environment variables."""
        limits = ProcessingLimits()
        return cls(
            environment=get_environment(),
            is_apify=is_apify_environment(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            
            groq_api_key=os.getenv("GROQ_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            
            max_retries=int(os.getenv("MAX_RETRIES", str(limits.MAX_RETRIES))),
            code_timeout_seconds=int(os.getenv("CODE_TIMEOUT_SECONDS", str(limits.CODE_TIMEOUT_SECONDS))),
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", str(limits.MAX_MEMORY_MB))),
            streaming_threshold_bytes=int(os.getenv("STREAMING_THRESHOLD_BYTES", str(limits.STREAMING_THRESHOLD_BYTES))),
            streaming_chunk_size=int(os.getenv("STREAMING_CHUNK_SIZE", str(limits.STREAMING_CHUNK_SIZE))),
            
            output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        )
    
    def get_api_key(self, provider: LLMProvider) -> str | None:
        """Get API key for a specific provider."""
        key_map = {
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GOOGLE: self.google_api_key,
        }
        return key_map.get(provider)


# =============================================================================
# RUNTIME CONFIGURATION (from Actor input)
# =============================================================================

@dataclass(slots=True)
class RuntimeConfig:
    """
    Runtime configuration derived from Actor input.
    Mutable during initialization phase only.
    """
    
    # Required
    prompt: str
    data_source: str  # URL or path
    
    # LLM
    provider: LLMProvider
    api_key: str | None
    
    # Processing
    output_format: OutputFormat
    include_generated_code: bool
    max_retries: int
    streaming_mode: bool
    
    @classmethod
    def from_actor_input(cls, actor_input: dict[str, Any], config: Config) -> RuntimeConfig:
        """Create runtime config from Actor input and base config."""
        
        # Determine data source
        data_source = actor_input.get("datasetUrl") or actor_input.get("datasetPath", "")
        if not data_source:
            # Check for multiple URLs
            urls = actor_input.get("datasetUrls", [])
            if urls:
                data_source = urls[0] if isinstance(urls, list) else urls
        
        if not data_source:
            raise ValueError("Data source is required (datasetUrl, datasetUrls, or datasetPath)")
        
        # Parse provider
        provider_str = actor_input.get("llmProvider", "groq").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise ValueError(f"Invalid LLM provider: {provider_str}. Valid options: {[p.value for p in LLMProvider]}")
        
        # Get API key (input takes precedence over env)
        api_key = cls._resolve_api_key(actor_input, provider, config)
        
        # Parse output format
        output_format_str = actor_input.get("outputFormat", "csv").lower()
        try:
            output_format = OutputFormat(output_format_str)
        except ValueError:
            raise ValueError(f"Invalid output format: {output_format_str}. Valid options: {[f.value for f in OutputFormat]}")
        
        return cls(
            prompt=actor_input.get("prompt", ""),
            data_source=data_source,
            provider=provider,
            api_key=api_key,
            output_format=output_format,
            include_generated_code=actor_input.get("includeGeneratedCode", True),
            max_retries=actor_input.get("maxRetries", config.max_retries),
            streaming_mode=actor_input.get("streamingMode", False),
        )
    
    @staticmethod
    def _resolve_api_key(
        actor_input: dict[str, Any],
        provider: LLMProvider,
        config: Config,
    ) -> str | None:
        """Resolve API key from input or environment."""
        # Try input field first
        input_field = API_KEY_INPUT_FIELDS.get(provider)
        if input_field:
            key = actor_input.get(input_field)
            if key:
                return key
        
        # Fall back to environment
        return config.get_api_key(provider)
    
    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors."""
        errors: list[str] = []
        
        if not self.prompt:
            errors.append("Prompt is required")
        
        if not self.data_source:
            errors.append("Data source (datasetUrl or datasetPath) is required")
        
        # API key required for non-Groq providers (Groq has free tier)
        if self.provider != LLMProvider.GROQ and not self.api_key:
            errors.append(f"API key required for provider: {self.provider.value}")
        
        return errors


# =============================================================================
# SINGLETON CONFIG ACCESSOR
# =============================================================================

@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get cached configuration instance.
    Config is immutable so caching is safe.
    """
    return Config.from_environment()


def reset_config() -> None:
    """Reset config cache. Useful for testing."""
    get_config.cache_clear()
