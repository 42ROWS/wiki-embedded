"""
Custom exceptions with structured error handling.
Each exception carries context for debugging and user-friendly messages.
"""
from dataclasses import dataclass, field
from typing import Any

from polars_runner.core.constants import ErrorCategory


@dataclass
class ErrorContext:
    """Structured context for errors."""
    category: ErrorCategory
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    suggestion: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "suggestion": self.suggestion,
        }


class TransformerError(Exception):
    """Base exception for AI Data Transformer."""
    
    def __init__(self, context: ErrorContext):
        self.context = context
        super().__init__(context.message)
    
    @property
    def user_message(self) -> str:
        """User-friendly error message."""
        msg = self.context.message
        if self.context.suggestion:
            msg += f" Suggestion: {self.context.suggestion}"
        return msg
    
    def to_dict(self) -> dict[str, Any]:
        return self.context.to_dict()


class ValidationError(TransformerError):
    """Input validation failed."""
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        suggestion: str | None = None,
    ):
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            message=message,
            details={"field": field, "value": str(value)[:100] if value else None},
            recoverable=True,
            suggestion=suggestion or "Please check your input and try again.",
        )
        super().__init__(context)


class DataLoadingError(TransformerError):
    """Failed to load data from source."""
    
    def __init__(
        self,
        message: str,
        source: str | None = None,
        original_error: Exception | None = None,
    ):
        context = ErrorContext(
            category=ErrorCategory.DATA_LOADING,
            message=message,
            details={
                "source": source,
                "original_error": str(original_error) if original_error else None,
            },
            recoverable=True,
            suggestion="Check that the file URL is accessible and the format is correct.",
        )
        super().__init__(context)


class SchemaMismatchError(TransformerError):
    """Schema mismatch when merging multiple files."""
    
    def __init__(
        self,
        message: str,
        expected_schema: dict[str, str] | None = None,
        actual_schema: dict[str, str] | None = None,
        file_name: str | None = None,
    ):
        context = ErrorContext(
            category=ErrorCategory.SCHEMA_MISMATCH,
            message=message,
            details={
                "expected_schema": expected_schema,
                "actual_schema": actual_schema,
                "file_name": file_name,
            },
            recoverable=True,
            suggestion="Ensure all files have compatible column names and types.",
        )
        super().__init__(context)


class LLMGenerationError(TransformerError):
    """LLM failed to generate valid code."""
    
    def __init__(
        self,
        message: str,
        provider: str | None = None,
        prompt_preview: str | None = None,
        attempts: int = 0,
    ):
        context = ErrorContext(
            category=ErrorCategory.LLM_GENERATION,
            message=message,
            details={
                "provider": provider,
                "prompt_preview": prompt_preview[:200] if prompt_preview else None,
                "attempts": attempts,
            },
            recoverable=True,
            suggestion="Try rephrasing your transformation prompt with more specific column names.",
        )
        super().__init__(context)


class CodeExecutionError(TransformerError):
    """Generated code failed to execute."""
    
    def __init__(
        self,
        message: str,
        code: str | None = None,
        error_line: int | None = None,
        original_error: Exception | None = None,
    ):
        # Extract relevant code snippet around error
        code_snippet = None
        if code and error_line:
            lines = code.split("\n")
            start = max(0, error_line - 3)
            end = min(len(lines), error_line + 2)
            code_snippet = "\n".join(lines[start:end])
        
        context = ErrorContext(
            category=ErrorCategory.CODE_EXECUTION,
            message=message,
            details={
                "code_snippet": code_snippet,
                "error_line": error_line,
                "original_error": str(original_error) if original_error else None,
            },
            recoverable=True,
            suggestion="The generated code had an error. Try simplifying your request.",
        )
        super().__init__(context)


class SecurityError(TransformerError):
    """Security violation in generated code."""
    
    def __init__(
        self,
        message: str,
        violation_type: str | None = None,
        code_fragment: str | None = None,
    ):
        context = ErrorContext(
            category=ErrorCategory.CODE_EXECUTION,
            message=message,
            details={
                "violation_type": violation_type,
                "code_fragment": code_fragment[:100] if code_fragment else None,
            },
            recoverable=False,
            suggestion=None,
        )
        super().__init__(context)


class TimeoutError(TransformerError):
    """Operation timed out."""
    
    def __init__(
        self,
        message: str,
        operation: str | None = None,
        timeout_seconds: int | None = None,
    ):
        context = ErrorContext(
            category=ErrorCategory.TIMEOUT,
            message=message,
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
            },
            recoverable=True,
            suggestion="Try with a smaller dataset or simpler transformation.",
        )
        super().__init__(context)


class MemoryError(TransformerError):
    """Out of memory during processing."""
    
    def __init__(
        self,
        message: str,
        estimated_size_mb: float | None = None,
        available_mb: float | None = None,
    ):
        context = ErrorContext(
            category=ErrorCategory.MEMORY,
            message=message,
            details={
                "estimated_size_mb": estimated_size_mb,
                "available_mb": available_mb,
            },
            recoverable=True,
            suggestion="Enable streaming mode for large files or reduce dataset size.",
        )
        super().__init__(context)
