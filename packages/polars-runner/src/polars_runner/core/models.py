"""
Data models - Typed dataclasses for input/output.
Immutable where possible for thread safety.
"""
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

from polars_runner.core.constants import (
    LLMProvider,
    OutputFormat,
    InputFormat,
    DataSourceType,
    ExecutionStatus,
)


# =============================================================================
# INPUT MODELS
# =============================================================================

@dataclass(frozen=True)
class DataSource:
    """Single data source (file or dataset)."""
    source_type: DataSourceType
    location: str  # URL, file path, or dataset ID
    format: InputFormat | None = None  # Auto-detected if None
    inline_data: Any = None  # Direct JSON data (for INLINE type)
    table_name: str | None = None  # Named table for multi-table inputData

    def __post_init__(self):
        # Infer format from URL/path if not provided
        if self.format is None and self.location:
            ext = self.location.split(".")[-1].lower().split("?")[0]
            try:
                object.__setattr__(self, "format", InputFormat.from_extension(ext))
            except ValueError:
                pass  # Will be detected during loading


@dataclass(frozen=True)
class TransformationInput:
    """Complete input for transformation."""
    # Required
    prompt: str
    data_sources: tuple[DataSource, ...]
    
    # LLM configuration
    llm_provider: LLMProvider = LLMProvider.GROQ
    api_key: str | None = None
    fallback_provider: LLMProvider | None = None
    fallback_api_key: str | None = None
    
    # Output configuration
    output_format: OutputFormat = OutputFormat.CSV
    include_generated_code: bool = True
    
    # Execution options
    max_retries: int = 3
    enable_streaming: bool = False  # Auto-enabled for large files
    use_advanced_features: bool = False  # Premium tier: Gemini Pro + RAG
    
    @classmethod
    def from_actor_input(cls, actor_input: dict[str, Any]) -> "TransformationInput":
        """Factory from Apify Actor input."""
        # Parse data sources
        sources = []

        # Inline JSON data (highest priority - zero I/O)
        if inline_data := actor_input.get("inputData"):
            if isinstance(inline_data, dict):
                # Check if it's multi-table: {"table1": [...], "table2": [...]}
                first_value = next(iter(inline_data.values()), None)
                if isinstance(first_value, list):
                    # Multi-table format - create separate DataSource for each table
                    for table_name, table_data in inline_data.items():
                        sources.append(DataSource(
                            source_type=DataSourceType.INLINE,
                            location=f"inline:{table_name}",
                            inline_data=table_data,
                            table_name=table_name,
                        ))
                else:
                    # Single object: {"col": "val"} -> wrap in array
                    sources.append(DataSource(
                        source_type=DataSourceType.INLINE,
                        location="inline",
                        inline_data=[inline_data],
                    ))
            else:
                # Array format: [{...}, {...}]
                sources.append(DataSource(
                    source_type=DataSourceType.INLINE,
                    location="inline",
                    inline_data=inline_data,
                ))

        # URL-based sources
        if urls := actor_input.get("datasetUrls"):
            for url in urls if isinstance(urls, list) else [urls]:
                sources.append(DataSource(
                    source_type=DataSourceType.URL,
                    location=url,
                ))
        
        # Uploaded files
        if uploads := actor_input.get("uploadedFiles"):
            for upload_url in uploads if isinstance(uploads, list) else [uploads]:
                sources.append(DataSource(
                    source_type=DataSourceType.UPLOAD,
                    location=upload_url,
                ))
        
        # Apify dataset
        if dataset_id := actor_input.get("apifyDatasetId"):
            sources.append(DataSource(
                source_type=DataSourceType.APIFY_DATASET,
                location=dataset_id,
            ))

        # Simple alias: datasetId (user-friendly for integrations/API)
        if dataset_id := actor_input.get("datasetId"):
            sources.append(DataSource(
                source_type=DataSourceType.APIFY_DATASET,
                location=dataset_id,
            ))

        # Webhook trigger: resource.defaultDatasetId
        # When Actor is triggered by webhook/integration from another Actor,
        # the payload contains resource.defaultDatasetId with the previous run's dataset
        if resource := actor_input.get("resource"):
            if isinstance(resource, dict):
                if webhook_dataset_id := resource.get("defaultDatasetId"):
                    sources.append(DataSource(
                        source_type=DataSourceType.APIFY_DATASET,
                        location=webhook_dataset_id,
                    ))

        # Single URL (backward compatibility)
        if not sources and (url := actor_input.get("datasetUrl")):
            sources.append(DataSource(
                source_type=DataSourceType.URL,
                location=url,
            ))
        
        return cls(
            prompt=actor_input.get("prompt", ""),
            data_sources=tuple(sources),
            llm_provider=LLMProvider(actor_input.get("llmProvider", "groq")),
            api_key=_get_api_key(actor_input, actor_input.get("llmProvider", "groq")),
            fallback_provider=LLMProvider(fp) if (fp := actor_input.get("fallbackProvider")) else None,
            fallback_api_key=_get_api_key(actor_input, actor_input.get("fallbackProvider")),
            output_format=OutputFormat(actor_input.get("outputFormat", "csv")),
            include_generated_code=actor_input.get("includeGeneratedCode", True),
            max_retries=actor_input.get("maxRetries", 3),
            enable_streaming=actor_input.get("enableStreaming", False),
            use_advanced_features=actor_input.get("useAdvancedFeatures", False),
        )


def _get_api_key(actor_input: dict, provider: str | None) -> str | None:
    """Extract API key for provider from input."""
    if not provider:
        return None
    key_mapping = {
        "groq": "groqApiKey",
        "anthropic": "anthropicApiKey",
        "openai": "openaiApiKey",
        "google": "googleApiKey",
    }
    return actor_input.get(key_mapping.get(provider, ""))


# =============================================================================
# OUTPUT MODELS
# =============================================================================

@dataclass
class SchemaInfo:
    """Dataset schema information."""
    columns: dict[str, str]  # column_name -> dtype
    row_count: int
    null_counts: dict[str, int]
    sample_values: dict[str, list[Any]]
    
    def to_llm_description(self) -> str:
        """Generate schema description for LLM prompt."""
        lines = []
        for col, dtype in self.columns.items():
            samples = self.sample_values.get(col, [])
            sample_str = str(samples[:3]) if samples else "[]"
            null_count = self.null_counts.get(col, 0)
            null_info = f", {null_count} nulls" if null_count > 0 else ""
            lines.append(f"- {col}: {dtype} (examples: {sample_str}{null_info})")
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "row_count": self.row_count,
            "null_counts": self.null_counts,
        }


@dataclass
class GenerationResult:
    """Result of code generation."""
    code: str
    provider_used: LLMProvider
    tokens_used: int
    generation_time_ms: int
    attempts: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider_used.value,
            "tokens_used": self.tokens_used,
            "generation_time_ms": self.generation_time_ms,
            "attempts": self.attempts,
        }


@dataclass
class TransformationResult:
    """Complete transformation result."""
    status: ExecutionStatus
    
    # Input info
    input_sources_count: int
    input_rows_total: int
    input_columns: list[str]
    
    # Output info
    output_rows: int
    output_columns: list[str]
    output_file: str
    output_preview: list[dict[str, Any]]

    # Execution info
    execution_time_ms: int
    generation_result: GenerationResult | None
    generated_code: str | None

    # Optional fields (with defaults)
    output_data: list[dict[str, Any]] | None = None  # Full data if < 10MB
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for Actor.push_data()."""
        result = {
            "status": self.status.value,
            "input_sources_count": self.input_sources_count,
            "input_rows_total": self.input_rows_total,
            "input_columns": self.input_columns,
            "output_rows": self.output_rows,
            "output_columns": self.output_columns,
            "output_file": self.output_file,
            "output_preview": self.output_preview,
            "execution_time_ms": self.execution_time_ms,
            "generation_info": self.generation_result.to_dict() if self.generation_result else None,
            "generated_code": self.generated_code,
            "warnings": self.warnings,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }
        # Include full data only if available (< 10MB limit)
        if self.output_data is not None:
            result["output_data"] = self.output_data
        return result


# =============================================================================
# INTERNAL MODELS
# =============================================================================

@dataclass
class LoadedDataset:
    """Result of loading a single data source."""
    source: DataSource
    schema: SchemaInfo
    row_count: int
    estimated_size_mb: float
    load_time_ms: int
    
    # The actual data is NOT stored here - use lazy loading
    # This is just metadata about the loaded data


@dataclass
class MergedDatasetInfo:
    """Information about merged datasets."""
    sources: tuple[LoadedDataset, ...]
    total_rows: int
    unified_schema: SchemaInfo
    merge_strategy: str  # "concat_vertical", "concat_horizontal", "join"
    warnings: list[str] = field(default_factory=list)
