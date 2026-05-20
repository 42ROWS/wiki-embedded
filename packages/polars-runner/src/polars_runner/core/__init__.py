"""Core module - constants, exceptions, models, types, executor, error_analyzer."""
from polars_runner.core.constants import (
    # Enums
    LLMProvider,
    LLMModel,
    OutputFormat,
    InputFormat,
    DataSourceType,
    ExecutionStatus,
    ErrorCategory,
    ExecutionErrorType,
    # Config classes
    LLMConfig,
    ExecutionLimits,
    ProcessingLimits,
    FilePatterns,
    PolarsConfig,
    APIEndpoints,
    SecurityConfig,
    # Singleton instances
    LLM_CONFIGS,
    LIMITS,
    FILE_PATTERNS,
    POLARS_CONFIG,
    API_ENDPOINTS,
    # Security
    BLOCKED_IMPORTS,
    BLOCKED_FUNCTIONS,
    ALLOWED_POLARS_NAMESPACES,
    # Mappings
    API_KEY_ENV_VARS,
    API_KEY_INPUT_FIELDS,
    EXTENSION_TO_FORMAT,
    DEFAULT_MODELS,
)
from polars_runner.core.exceptions import (
    TransformerError,
    ValidationError,
    DataLoadingError,
    SchemaMismatchError,
    LLMGenerationError,
    CodeExecutionError,
    SecurityError,
    TimeoutError,
    MemoryError,
    ErrorContext,
)
from polars_runner.core.models import (
    DataSource,
    TransformationInput,
    SchemaInfo,
    GenerationResult,
    TransformationResult,
    LoadedDataset,
    MergedDatasetInfo,
)
from polars_runner.core.types import (
    # Type aliases
    LazyFrameType,
    DataFrameType,
    ExprType,
    SchemaType,
    JsonValue,
    JsonDict,
    PathLike,
    # TypedDicts
    ActorInput,
    TransformResult,
    LLMResponse,
    # Protocols
    LLMClientProtocol,
    DataLoaderProtocol,
    DataExporterProtocol,
    CodeExecutorProtocol,
    # Dataclasses
    DatasetStats,
    ExecutionResult,
    TransformContext,
    ColumnInfo,
)
from polars_runner.core.executor import (
    CodeValidator,
    CodeExecutor,
    execute_transformation,
)
from polars_runner.core.error_analyzer import (
    ErrorAnalyzer,
    ErrorAnalysis,
    has_risky_join_pattern,
)

__all__ = [
    # Enums
    "LLMProvider",
    "LLMModel",
    "OutputFormat",
    "InputFormat",
    "DataSourceType",
    "ExecutionStatus",
    "ErrorCategory",
    "ExecutionErrorType",
    # Config classes
    "LLMConfig",
    "ExecutionLimits",
    "ProcessingLimits",
    "FilePatterns",
    "PolarsConfig",
    "APIEndpoints",
    "SecurityConfig",
    # Singleton instances
    "LLM_CONFIGS",
    "LIMITS",
    "FILE_PATTERNS",
    "POLARS_CONFIG",
    "API_ENDPOINTS",
    # Security
    "BLOCKED_IMPORTS",
    "BLOCKED_FUNCTIONS",
    "ALLOWED_POLARS_NAMESPACES",
    # Mappings
    "API_KEY_ENV_VARS",
    "API_KEY_INPUT_FIELDS",
    "EXTENSION_TO_FORMAT",
    "DEFAULT_MODELS",
    # Exceptions
    "TransformerError",
    "ValidationError",
    "DataLoadingError",
    "SchemaMismatchError",
    "LLMGenerationError",
    "CodeExecutionError",
    "SecurityError",
    "TimeoutError",
    "MemoryError",
    "ErrorContext",
    # Models
    "DataSource",
    "TransformationInput",
    "SchemaInfo",
    "GenerationResult",
    "TransformationResult",
    "LoadedDataset",
    "MergedDatasetInfo",
    # Types
    "LazyFrameType",
    "DataFrameType",
    "ExprType",
    "SchemaType",
    "JsonValue",
    "JsonDict",
    "PathLike",
    "ActorInput",
    "TransformResult",
    "LLMResponse",
    "LLMClientProtocol",
    "DataLoaderProtocol",
    "DataExporterProtocol",
    "CodeExecutorProtocol",
    "DatasetStats",
    "ExecutionResult",
    "TransformContext",
    "ColumnInfo",
    # Executor
    "CodeValidator",
    "CodeExecutor",
    "execute_transformation",
    # Error Analyzer
    "ErrorAnalyzer",
    "ErrorAnalysis",
    "has_risky_join_pattern",
]
