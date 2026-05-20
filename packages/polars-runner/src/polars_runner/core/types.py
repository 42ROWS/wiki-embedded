"""
Type definitions and protocols for strict type safety.
All custom types and interfaces defined here.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, TypedDict, runtime_checkable

import polars as pl

from .constants import LLMProvider, OutputFormat


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Polars types
LazyFrameType: TypeAlias = pl.LazyFrame
DataFrameType: TypeAlias = pl.DataFrame
ExprType: TypeAlias = pl.Expr
SchemaType: TypeAlias = dict[str, pl.DataType]

# JSON-serializable types
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]

# File paths
PathLike: TypeAlias = str


# =============================================================================
# TYPED DICTS FOR INPUT/OUTPUT
# =============================================================================

class ActorInput(TypedDict, total=False):
    """Actor input schema type."""
    # Required
    prompt: str
    
    # Data source (one required)
    datasetUrl: str
    datasetPath: str
    
    # Optional
    outputFormat: str
    llmProvider: str
    groqApiKey: str
    anthropicApiKey: str
    openaiApiKey: str
    googleApiKey: str
    includeGeneratedCode: bool
    maxRetries: int
    streamingMode: bool


class TransformResult(TypedDict):
    """Result of a transformation operation."""
    status: str
    input_rows: int
    output_rows: int
    columns: list[str]
    execution_time_ms: int
    llm_provider: str
    llm_model: str
    tokens_used: int
    generated_code: str | None
    output_file: str
    output_preview: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]


class SchemaInfo(TypedDict):
    """Schema information for LLM prompt."""
    column_name: str
    dtype: str
    sample_values: list[Any]
    null_count: int
    unique_count: int


class LLMResponse(TypedDict):
    """Response from LLM provider."""
    code: str
    tokens_input: int
    tokens_output: int
    model: str


# =============================================================================
# PROTOCOLS (Interfaces)
# =============================================================================

@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations."""
    
    @property
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        ...
    
    async def generate_code(
        self,
        user_prompt: str,
        schema_info: str,
        previous_error: str | None = None,
    ) -> LLMResponse:
        """Generate Polars transformation code."""
        ...


@runtime_checkable  
class DataLoaderProtocol(Protocol):
    """Protocol for data loading implementations."""
    
    def load_lazy(self, source: PathLike) -> LazyFrameType:
        """Load data as LazyFrame for streaming."""
        ...
    
    def get_schema_info(self, lf: LazyFrameType, sample_rows: int) -> list[SchemaInfo]:
        """Extract schema information for LLM."""
        ...


@runtime_checkable
class DataExporterProtocol(Protocol):
    """Protocol for data export implementations."""
    
    def export(
        self,
        lf: LazyFrameType,
        output_format: OutputFormat,
        output_path: PathLike,
        streaming: bool = True,
    ) -> int:
        """Export data and return row count."""
        ...


@runtime_checkable
class CodeExecutorProtocol(Protocol):
    """Protocol for code execution implementations."""
    
    def execute(
        self,
        code: str,
        df: DataFrameType,
    ) -> DataFrameType:
        """Execute generated code on DataFrame."""
        ...
    
    def validate(self, code: str) -> tuple[bool, list[str]]:
        """Validate code safety. Returns (is_valid, errors)."""
        ...


# =============================================================================
# DATA CLASSES FOR STRUCTURED DATA
# =============================================================================

@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Statistics about a dataset."""
    row_count: int
    column_count: int
    estimated_size_bytes: int
    columns: tuple[str, ...]
    
    @property
    def should_stream(self) -> bool:
        """Determine if streaming should be used."""
        from .constants import ProcessingLimits
        return self.estimated_size_bytes > ProcessingLimits.STREAMING_THRESHOLD_BYTES


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result of code execution."""
    success: bool
    result_df: DataFrameType | None
    error_message: str | None
    execution_time_ms: int
    
    @classmethod
    def from_success(cls, df: DataFrameType, time_ms: int) -> "ExecutionResult":
        return cls(success=True, result_df=df, error_message=None, execution_time_ms=time_ms)
    
    @classmethod
    def from_error(cls, error: str, time_ms: int) -> "ExecutionResult":
        return cls(success=False, result_df=None, error_message=error, execution_time_ms=time_ms)


@dataclass(slots=True)
class TransformContext:
    """Context for a transformation operation."""
    prompt: str
    provider: LLMProvider
    output_format: OutputFormat
    max_retries: int
    include_code: bool
    streaming_mode: bool
    warnings: list[str] = field(default_factory=list)
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """Information about a single column."""
    name: str
    dtype: pl.DataType
    sample_values: tuple[Any, ...]
    null_count: int
    unique_count: int
    
    def to_prompt_string(self) -> str:
        """Format column info for LLM prompt."""
        samples = ", ".join(repr(v) for v in self.sample_values[:3])
        return f"- {self.name}: {self.dtype} (examples: [{samples}], nulls: {self.null_count})"
