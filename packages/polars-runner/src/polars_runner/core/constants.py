"""
Centralized constants and enums - ZERO magic values in codebase.
All configurable values defined here for maintainability.
"""
import os
from enum import Enum, auto
from dataclasses import dataclass
from typing import Final


# =============================================================================
# ENUMS
# =============================================================================

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GOOGLE_FLASH_LITE = "google_flash_lite"  # Hosted mode - basic tier
    GOOGLE_PRO = "google_pro"  # Hosted mode - premium tier with reasoning
    
    @property
    def display_name(self) -> str:
        return _LLM_DISPLAY_NAMES[self]
    
    @property
    def requires_api_key(self) -> bool:
        # Flash Lite is our hosted provider - doesn't require user API key
        return self != LLMProvider.GOOGLE_FLASH_LITE
    
    @property
    def model_id(self) -> str:
        return _LLM_MODEL_IDS[self]
    
    @property
    def env_var_name(self) -> str:
        if self == LLMProvider.GOOGLE_FLASH_LITE:
            return "GEMINI_HOSTED_API_KEY"  # Our internal API key
        return f"{self.value.upper()}_API_KEY"
    
    @property
    def is_hosted(self) -> bool:
        """Check if this is a hosted (non-BYOK) provider."""
        return self == LLMProvider.GOOGLE_FLASH_LITE


class OutputFormat(str, Enum):
    """Supported output formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "xlsx"
    
    @property
    def content_type(self) -> str:
        return _OUTPUT_CONTENT_TYPES[self]
    
    @property
    def file_extension(self) -> str:
        return self.value


class InputFormat(str, Enum):
    """Supported input file formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "xlsx"
    EXCEL_OLD = "xls"
    
    @classmethod
    def from_extension(cls, ext: str) -> "InputFormat":
        ext = ext.lower().lstrip(".")
        mapping = {
            "csv": cls.CSV,
            "json": cls.JSON,
            "parquet": cls.PARQUET,
            "xlsx": cls.EXCEL,
            "xls": cls.EXCEL_OLD,
        }
        if ext not in mapping:
            raise ValueError(f"Unsupported file extension: {ext}")
        return mapping[ext]


class DataSourceType(str, Enum):
    """Type of data source."""
    URL = "url"
    UPLOAD = "upload"
    APIFY_DATASET = "apify_dataset"
    INLINE = "inline"  # Direct JSON data in request


class ExecutionStatus(str, Enum):
    """Actor execution status."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class ErrorCategory(str, Enum):
    """Error categories for structured error handling."""
    VALIDATION = "validation"
    DATA_LOADING = "data_loading"
    SCHEMA_MISMATCH = "schema_mismatch"
    LLM_GENERATION = "llm_generation"
    CODE_EXECUTION = "code_execution"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    UNKNOWN = "unknown"


class ExecutionErrorType(str, Enum):
    """
    Specific error types for code execution failures.
    Used by ErrorAnalyzer to provide targeted recovery suggestions.
    """
    # Column errors
    COLUMN_NOT_FOUND = "column_not_found"
    JOIN_RIGHT_KEY_DROPPED = "join_right_key_dropped"  # Polars drops right key in JOIN
    CASE_SENSITIVITY = "case_sensitivity"  # Wrong column case

    # Type errors
    TYPE_MISMATCH = "type_mismatch"
    CAST_ERROR = "cast_error"

    # Syntax/structure errors
    PANDAS_SYNTAX = "pandas_syntax"  # Used Pandas instead of Polars
    INVALID_OPERATION = "invalid_operation"
    MISSING_RESULT = "missing_result"

    # Other
    UNKNOWN = "unknown"


# =============================================================================
# PRICING EVENTS - Apify Pay-per-Event
# =============================================================================

class PricingEvent(str, Enum):
    """Apify pricing events for pay-per-event model."""
    TRANSFORMATION_BYOK = "transformation_byok"      # User provides API key
    TRANSFORMATION_BASIC = "transformation_basic"    # Hosted Flash-Lite (basic tier)
    TRANSFORMATION_PREMIUM = "transformation_premium"  # Hosted Gemini Pro (premium tier)


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

_LLM_DISPLAY_NAMES: dict["LLMProvider", str] = {
    LLMProvider.GROQ: "⚡ Groq Llama 3.3 70B (FREE & Fast)",
    LLMProvider.ANTHROPIC: "🧠 Claude Sonnet 4 (Best for code)",
    LLMProvider.OPENAI: "🌐 GPT-4o (Reliable)",
    LLMProvider.GOOGLE: "💎 Gemini 2.0 Flash (Fast)",
    LLMProvider.GOOGLE_FLASH_LITE: "🚀 Gemini 2.5 Flash-Lite (Basic Hosted)",
    LLMProvider.GOOGLE_PRO: "💎 Gemini 2.5 Pro (Premium Hosted with Reasoning)",
}

_LLM_MODEL_IDS: dict["LLMProvider", str] = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.GOOGLE: "gemini-2.0-flash",
    LLMProvider.GOOGLE_FLASH_LITE: "gemini-2.5-flash-lite",
    LLMProvider.GOOGLE_PRO: "gemini-2.5-pro",
}

_OUTPUT_CONTENT_TYPES: dict[OutputFormat, str] = {
    OutputFormat.CSV: "text/csv",
    OutputFormat.JSON: "application/json",
    OutputFormat.PARQUET: "application/octet-stream",
    OutputFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""
    max_tokens: int = 4096
    temperature: float = 0.0  # Deterministic for code generation
    timeout_seconds: int = 60


# Provider-specific configs
LLM_CONFIGS: dict["LLMProvider", LLMConfig] = {
    LLMProvider.GROQ: LLMConfig(max_tokens=4096, timeout_seconds=30),
    LLMProvider.ANTHROPIC: LLMConfig(max_tokens=4096, timeout_seconds=60),
    LLMProvider.OPENAI: LLMConfig(max_tokens=4096, timeout_seconds=60),
    LLMProvider.GOOGLE: LLMConfig(max_tokens=4096, timeout_seconds=45),
    LLMProvider.GOOGLE_FLASH_LITE: LLMConfig(max_tokens=4096, timeout_seconds=30),
    LLMProvider.GOOGLE_PRO: LLMConfig(max_tokens=8192, timeout_seconds=60),  # Premium: more tokens, longer timeout
}


# =============================================================================
# EXECUTION LIMITS
# =============================================================================

@dataclass(frozen=True)
class ExecutionLimits:
    """Execution limits for safety and performance."""
    # Code execution
    max_execution_seconds: int = 300  # 5 minutes max
    max_memory_mb: int = 2048  # 2GB max for code execution
    max_retries: int = 3
    
    # Data processing
    max_file_size_mb: int = 500
    max_files_count: int = 20
    max_total_rows: int = 10_000_000  # 10M rows
    max_columns: int = 500
    
    # Streaming thresholds
    streaming_threshold_rows: int = 1_000_000  # Use streaming above 1M rows
    streaming_threshold_mb: int = 100  # Use streaming above 100MB
    chunk_size: int = 100_000  # Rows per chunk for streaming
    
    # Preview limits
    preview_rows: int = 10
    schema_sample_rows: int = 1000  # Rows to sample for schema inference

    # Output data limits
    max_output_data_bytes: int = 10 * 1024 * 1024  # 10MB limit for output_data in response


LIMITS: Final[ExecutionLimits] = ExecutionLimits()


# =============================================================================
# FILE PATTERNS
# =============================================================================

@dataclass(frozen=True)
class FilePatterns:
    """File naming patterns."""
    output_data: str = "transformed_data"
    generated_code: str = "generated_code.py"
    execution_log: str = "execution_log.txt"
    schema_info: str = "schema_info.json"


FILE_PATTERNS: Final[FilePatterns] = FilePatterns()


# =============================================================================
# POLARS CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class PolarsConfig:
    """Polars-specific configuration."""
    # Inference settings
    infer_schema_length: int = 10000
    null_values: tuple[str, ...] = ("", "null", "NULL", "None", "NA", "N/A", "n/a", "-")
    
    # CSV parsing
    csv_separator_detect: bool = True
    csv_encoding_detect: bool = True
    
    # Performance
    rechunk_after_concat: bool = True
    use_streaming_threshold_mb: int = 100


POLARS_CONFIG: Final[PolarsConfig] = PolarsConfig()


# =============================================================================
# API ENDPOINTS (for reference/validation)
# =============================================================================

@dataclass(frozen=True)
class APIEndpoints:
    """External API endpoints."""
    anthropic: str = "https://api.anthropic.com/v1/messages"
    openai: str = "https://api.openai.com/v1/chat/completions"
    groq: str = "https://api.groq.com/openai/v1/chat/completions"
    google: str = "https://generativelanguage.googleapis.com/v1beta/models"


API_ENDPOINTS: Final[APIEndpoints] = APIEndpoints()


# =============================================================================
# PROCESSING LIMITS
# =============================================================================

class ProcessingLimits:
    """Processing limits class for config module compatibility."""
    MAX_RETRIES: Final[int] = 3
    CODE_TIMEOUT_SECONDS: Final[int] = 300
    MAX_MEMORY_MB: Final[int] = 2048
    STREAMING_THRESHOLD_BYTES: Final[int] = 100 * 1024 * 1024  # 100MB
    STREAMING_CHUNK_SIZE: Final[int] = 100_000
    SCHEMA_SAMPLE_ROWS: Final[int] = 1000
    MAX_PREVIEW_ROWS: Final[int] = 10
    MAX_TOKENS_OUTPUT: Final[int] = 4096


# =============================================================================
# API KEY MAPPINGS
# =============================================================================

API_KEY_ENV_VARS: Final[dict["LLMProvider", str]] = {
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
    LLMProvider.GOOGLE_FLASH_LITE: "GEMINI_HOSTED_API_KEY",
    LLMProvider.GOOGLE_PRO: "GEMINI_PRO_API_KEY",
}

API_KEY_INPUT_FIELDS: Final[dict["LLMProvider", str]] = {
    LLMProvider.GROQ: "groqApiKey",
    LLMProvider.ANTHROPIC: "anthropicApiKey",
    LLMProvider.OPENAI: "openaiApiKey",
    LLMProvider.GOOGLE: "googleApiKey",
    LLMProvider.GOOGLE_FLASH_LITE: "",  # Not user-provided
}


# =============================================================================
# FILE FORMAT MAPPINGS
# =============================================================================

EXTENSION_TO_FORMAT: Final[dict[str, InputFormat]] = {
    ".csv": InputFormat.CSV,
    ".json": InputFormat.JSON,
    ".parquet": InputFormat.PARQUET,
    ".xlsx": InputFormat.EXCEL,
    ".xls": InputFormat.EXCEL_OLD,
}


# =============================================================================
# DEFAULT MODELS
# =============================================================================

class LLMModel(str, Enum):
    """Available LLM models."""
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    GPT_4O = "gpt-4o"
    GEMINI_2_FLASH = "gemini-2.0-flash"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"


DEFAULT_MODELS: Final[dict["LLMProvider", LLMModel]] = {
    LLMProvider.GROQ: LLMModel.LLAMA_3_3_70B,
    LLMProvider.ANTHROPIC: LLMModel.CLAUDE_SONNET_4,
    LLMProvider.OPENAI: LLMModel.GPT_4O,
    LLMProvider.GOOGLE: LLMModel.GEMINI_2_FLASH,
    LLMProvider.GOOGLE_FLASH_LITE: LLMModel.GEMINI_25_FLASH_LITE,
}


# =============================================================================
# SECURITY CONFIG
# =============================================================================

@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration for code validation."""
    BLOCKED_PATTERNS: tuple[str, ...] = (
        "os.system",
        "subprocess",
        "exec(",
        "eval(",
        "compile(",
        "__import__",
        "open(",
        "file(",
        "input(",
        "breakpoint(",
        "pdb",
        "importlib",
        "pickle",
        "shelve",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "shutil",
        "pathlib",
        "glob",
        "tempfile",
    )
    
    ALLOWED_IMPORTS: tuple[str, ...] = (
        "polars",
        "pl",
        "datetime",
        "math",
        "re",
        "json",
        "typing",
    )
    
    ALLOWED_POLARS_METHODS: tuple[str, ...] = (
        "select", "filter", "with_columns", "group_by", "agg",
        "sort", "head", "tail", "unique", "drop_nulls", "fill_null",
        "join", "concat", "lazy", "collect", "alias", "cast",
        "when", "then", "otherwise", "over", "sum", "mean", "count",
        "min", "max", "first", "last", "len", "n_unique", "std", "var",
        "str", "dt", "list", "struct", "arr",
    )


# =============================================================================
# RAG CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) configuration."""
    # Pinecone settings
    similarity_threshold: float = 0.85  # Minimum similarity to reuse code
    top_k_results: int = 2  # Number of similar codes to show

    # Namespaces
    namespace_success: str = "code-success"
    namespace_failures: str = "code-failures"

    # Retrieval quality gate — vectors below this `quality_score` are filtered
    # out at `search_similar` time. Legacy vectors written before the semantic
    # validator existed score 50-70 (success+output+attempts+speed); validator-
    # gated vectors score 95+ thanks to the +30 validator_passed bonus.
    # Threshold 50 keeps legacy hits while strongly preferring validated code.
    min_quality_for_retrieval: float = 50.0

    # Storage limits (aligned with FREE tier)
    # With 1024 dimensions: 2GB / 1024 / 4 bytes / 1.5 (overhead) ≈ 341K vectors capacity
    # We use conservative 6.5K limit (1.9% of FREE tier capacity)
    max_vectors: int = 6500  # Self-imposed limit (1.9% of FREE tier)
    cleanup_threshold: int = 5850  # Cleanup at 90%
    cleanup_percentage: float = 0.10  # Remove bottom 10% by value score

    # Embedding model (Pinecone FREE inference)
    # llama-text-embed-v2: Superior performance, 12x faster than OpenAI, FREE tier
    # Supports: 384, 512, 768, 1024 (default), 2048 dimensions
    embedding_model: str = "llama-text-embed-v2"
    embedding_dimension: int = 1024  # Using 1024 for optimal storage

    # Quality scoring weights
    quality_weight_success: float = 0.4
    quality_weight_reuse: float = 6.0
    quality_max_reuse_score: int = 60

    # Metadata limits (Pinecone max is 40KB per vector)
    # We use slightly lower to leave room for system fields
    max_metadata_bytes: int = 38_000  # ~38KB max for user metadata
    max_prompt_chars: int = 8_000  # Generous limit for prompts
    max_code_chars: int = 15_000  # Generous limit for code
    max_error_chars: int = 1_000  # Error messages
    max_columns_in_metadata: int = 100  # Schema columns to store


RAG_CONFIG: Final[RAGConfig] = RAGConfig()


@dataclass(frozen=True)
class OracleConfig:
    """Configuration for the property-based testing oracle pipeline.

    See :mod:`polars_runner.executor.oracle`. The oracle is the project's
    **default** quality gate, not an optional feature; the env vars below
    exist only so an operator can *disable* or *soften* parts of the
    pipeline for debugging or offline runs without a premium LLM key.
    """
    # Master switch — kept on by default. Set ``POLARS_RUNNER_DISABLE_ORACLE=1``
    # to fall back to the legacy single-shot validator-only path.
    enabled: bool = True

    # Model used to extract the prompt-derived property contract. Premium
    # models (Gemini 2.5 Pro, Claude Sonnet 4) give markedly better oracles
    # than basic tiers — paper "From Prompts to Properties" (FSE 2024).
    oracle_model: str = "gemini-2.5-pro"
    # Pro runs in mandatory thinking mode: the internal reasoning consumes
    # tokens from this budget before any output is produced. 4096 is the
    # smallest value that empirically yields a non-empty JSON for typical
    # data-analysis prompts; below that, Pro returns "".
    oracle_max_tokens: int = 4096
    oracle_temperature: float = 0.0

    # When the oracle fails on the chosen output, treat the run as a semantic
    # failure and engage the existing retry loop. Set
    # ``POLARS_RUNNER_ORACLE_ADVISORY=1`` to keep the oracle non-blocking (it
    # is logged but the result is returned).
    oracle_is_blocking: bool = True


def _disabled(var: str) -> bool:
    """Return ``True`` iff the given env var is set to an explicit "off" value."""
    return os.getenv(var, "").lower() in ("0", "false", "no", "off")


ORACLE_CONFIG: Final[OracleConfig] = OracleConfig(
    enabled=not _disabled("POLARS_RUNNER_DISABLE_ORACLE"),
    oracle_is_blocking=not _disabled("POLARS_RUNNER_ORACLE_ADVISORY"),
)


# =============================================================================
# SECURITY
# =============================================================================

# Blocked imports in generated code
BLOCKED_IMPORTS: Final[frozenset[str]] = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "importlib",
    "builtins",
    "pickle",
    "shelve",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
})

# Blocked function calls
BLOCKED_FUNCTIONS: Final[frozenset[str]] = frozenset({
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
})

# Allowed Polars namespaces
ALLOWED_POLARS_NAMESPACES: Final[frozenset[str]] = frozenset({
    "pl",
    "polars",
})
