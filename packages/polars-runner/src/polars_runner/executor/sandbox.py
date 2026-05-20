"""
Sandboxed code execution with resource limits.
Executes validated Polars code safely.
"""
from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from typing import Any, Final

import polars as pl

from polars_runner.core.constants import ProcessingLimits
from polars_runner.core.types import DataFrameType, LazyFrameType, ExecutionResult
from .validator import (
    ValidationResult,
    extract_code_from_response,
    normalize_code,
    validate_code,
)


# Singleton limits instance
_LIMITS: Final[ProcessingLimits] = ProcessingLimits()


# =============================================================================
# TIMEOUT HANDLING
# =============================================================================

class ExecutionTimeout(Exception):
    """Raised when code execution times out."""
    pass


class ExecutionError(Exception):
    """Raised when code execution fails."""
    pass


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for execution timeout.
    
    Note: Only works on Unix systems. On Windows, timeout is ignored.
    
    Args:
        seconds: Maximum execution time
        
    Raises:
        ExecutionTimeout: If execution exceeds timeout
    """
    def handler(signum: int, frame: Any) -> None:
        raise ExecutionTimeout(f"Execution timed out after {seconds} seconds")
    
    # Try to set signal handler (Unix only)
    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        # Windows or signal not available - no timeout
        yield


# =============================================================================
# EXECUTION NAMESPACE
# =============================================================================

def create_execution_namespace(df: DataFrameType) -> dict[str, Any]:
    """
    Create isolated namespace for code execution.
    
    Only exposes polars and the input DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Namespace dict for exec()
    """
    return {
        # Polars module
        "pl": pl,
        "polars": pl,
        
        # Input data
        "df": df,
        
        # Result placeholder
        "result": None,
        
        # Builtins (limited)
        "__builtins__": {
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "True": True,
            "False": False,
            "None": None,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "print": print,  # Allow print for debugging
        },
    }


# =============================================================================
# CODE EXECUTOR
# =============================================================================

class CodeExecutor:
    """
    Executes validated Polars code in a sandboxed environment.
    
    Features:
    - Validation before execution
    - Timeout protection
    - Isolated namespace
    - Result extraction
    """
    
    def __init__(
        self,
        timeout_seconds: int | None = None,
    ) -> None:
        self._timeout = timeout_seconds or _LIMITS.CODE_TIMEOUT_SECONDS
    
    def execute(
        self,
        code: str,
        df: DataFrameType,
        validate: bool = True,
    ) -> ExecutionResult:
        """
        Execute code on DataFrame.
        
        Args:
            code: Python/Polars code to execute
            df: Input DataFrame
            validate: Whether to validate code first
            
        Returns:
            ExecutionResult with success/failure and result DataFrame
        """
        start_time = time.perf_counter()
        
        # Extract and normalize code
        code = extract_code_from_response(code)
        code = normalize_code(code)
        
        # Validate if requested
        if validate:
            validation = validate_code(code)
            if not validation.is_valid:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                return ExecutionResult.from_error(
                    f"Validation failed: {'; '.join(validation.errors)}",
                    elapsed_ms,
                )
        
        # Create isolated namespace
        namespace = create_execution_namespace(df)
        
        # Execute with timeout
        try:
            with timeout_context(self._timeout):
                exec(code, namespace)
        except ExecutionTimeout as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return ExecutionResult.from_error(str(e), elapsed_ms)
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return ExecutionResult.from_error(f"Execution error: {type(e).__name__}: {e}", elapsed_ms)
        
        # Extract result
        result = namespace.get("result")
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        if result is None:
            return ExecutionResult.from_error(
                "Code did not produce a 'result' variable",
                elapsed_ms,
            )
        
        # Handle LazyFrame - collect it
        if isinstance(result, pl.LazyFrame):
            try:
                result = result.collect()
            except Exception as e:
                return ExecutionResult.from_error(
                    f"Failed to collect LazyFrame: {e}",
                    elapsed_ms,
                )
        
        if not isinstance(result, pl.DataFrame):
            return ExecutionResult.from_error(
                f"Result is not a DataFrame: {type(result).__name__}",
                elapsed_ms,
            )
        
        return ExecutionResult.from_success(result, elapsed_ms)
    
    def execute_lazy(
        self,
        code: str,
        lf: LazyFrameType,
        streaming: bool = True,
    ) -> ExecutionResult:
        """
        Execute code on LazyFrame with optional streaming.
        
        For large datasets, keeps operations lazy until final collect.
        
        Args:
            code: Python/Polars code to execute
            lf: Input LazyFrame
            streaming: Use streaming engine for collect
            
        Returns:
            ExecutionResult with result DataFrame
        """
        start_time = time.perf_counter()
        
        # Extract and normalize code
        code = extract_code_from_response(code)
        code = normalize_code(code)
        
        # Validate
        validation = validate_code(code)
        if not validation.is_valid:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return ExecutionResult.from_error(
                f"Validation failed: {'; '.join(validation.errors)}",
                elapsed_ms,
            )
        
        # Collect to DataFrame for execution
        # (code expects 'df' variable)
        try:
            if streaming:
                df = lf.collect(engine="streaming")
            else:
                df = lf.collect()
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return ExecutionResult.from_error(
                f"Failed to collect input data: {e}",
                elapsed_ms,
            )
        
        # Execute on collected DataFrame
        return self.execute(code, df, validate=False)


# =============================================================================
# SINGLETON EXECUTOR
# =============================================================================

_executor: CodeExecutor | None = None


def get_executor() -> CodeExecutor:
    """Get cached executor instance."""
    global _executor
    if _executor is None:
        _executor = CodeExecutor()
    return _executor


def execute_code(code: str, df: DataFrameType) -> ExecutionResult:
    """Convenience function to execute code."""
    return get_executor().execute(code, df)
