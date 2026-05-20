"""
Secure code executor with sandbox and validation.
Executes LLM-generated Polars code safely.
"""
import ast
import re
import time
import signal
from typing import Any
from contextlib import contextmanager

import polars as pl

from polars_runner.core.constants import (
    LIMITS,
    BLOCKED_IMPORTS,
    BLOCKED_FUNCTIONS,
    ALLOWED_POLARS_NAMESPACES,
)
from polars_runner.core.exceptions import (
    CodeExecutionError,
    SecurityError,
    TimeoutError,
)


# =============================================================================
# SECURITY VALIDATOR
# =============================================================================

class CodeValidator:
    """Validates generated code for security issues."""
    
    @classmethod
    def validate(cls, code: str) -> None:
        """
        Validate code is safe to execute.
        
        Raises:
            SecurityError: If code contains dangerous patterns
        """
        cls._check_blocked_patterns(code)
        cls._validate_ast(code)
    
    @classmethod
    def _check_blocked_patterns(cls, code: str) -> None:
        """Check for blocked string patterns."""
        code_lower = code.lower()
        
        # Check blocked imports
        for blocked in BLOCKED_IMPORTS:
            patterns = [
                f"import {blocked}",
                f"from {blocked}",
                f"__import__('{blocked}",
                f'__import__("{blocked}',
            ]
            for pattern in patterns:
                if pattern in code_lower:
                    raise SecurityError(
                        f"Blocked import detected: {blocked}",
                        violation_type="blocked_import",
                        code_fragment=pattern,
                    )
        
        # Check blocked function calls
        for blocked in BLOCKED_FUNCTIONS:
            # Match function call pattern
            pattern = rf"\b{blocked}\s*\("
            if re.search(pattern, code):
                raise SecurityError(
                    f"Blocked function detected: {blocked}",
                    violation_type="blocked_function",
                    code_fragment=blocked,
                )
    
    @classmethod
    def _validate_ast(cls, code: str) -> None:
        """Validate AST for dangerous constructs."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeExecutionError(
                f"Syntax error in generated code: {e}",
                code=code,
                error_line=e.lineno,
                original_error=e,
            )
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_POLARS_NAMESPACES:
                        # Allow polars and common safe modules
                        if alias.name not in {"datetime", "re", "math"}:
                            raise SecurityError(
                                f"Unauthorized import: {alias.name}",
                                violation_type="unauthorized_import",
                                code_fragment=alias.name,
                            )
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in ALLOWED_POLARS_NAMESPACES:
                    if node.module not in {"datetime", "re", "math"}:
                        raise SecurityError(
                            f"Unauthorized import from: {node.module}",
                            violation_type="unauthorized_import",
                            code_fragment=node.module,
                        )
            
            # Check for attribute access to dangerous modules
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in BLOCKED_IMPORTS:
                        raise SecurityError(
                            f"Access to blocked module: {node.value.id}",
                            violation_type="blocked_module_access",
                            code_fragment=f"{node.value.id}.{node.attr}",
                        )


# =============================================================================
# TIMEOUT CONTEXT MANAGER
# =============================================================================

@contextmanager
def timeout_context(seconds: int):
    """Context manager for execution timeout (Unix only).
    
    NOTE: Disabled by default because signal-based timeout interferes
    with Polars internal generators, causing 'generator didn't stop after throw()'
    errors. Apify platform has its own timeout mechanisms.
    """
    # Disabled: signal-based timeout causes issues with Polars generators
    # Just yield without setting any alarm
    yield
    return
    
    # Original implementation (disabled):
    # def timeout_handler(signum, frame):
    #     raise TimeoutError(...)


# =============================================================================
# CODE EXECUTOR
# =============================================================================

class CodeExecutor:
    """
    Executes validated Polars code in a restricted namespace.
    """
    
    def __init__(self, timeout_seconds: int = LIMITS.max_execution_seconds):
        self._timeout = timeout_seconds
    
    def execute(
        self,
        code: str,
        df: pl.DataFrame | pl.LazyFrame,
        extra_dataframes: dict[str, pl.DataFrame | pl.LazyFrame] | None = None,
    ) -> pl.DataFrame:
        """
        Execute Polars transformation code.

        Args:
            code: Validated Python code
            df: Input DataFrame (eager or lazy)
            extra_dataframes: Optional dict of additional named DataFrames
                             (for multi-table JOIN operations)

        Returns:
            Transformed DataFrame

        Raises:
            CodeExecutionError: If execution fails
            SecurityError: If code is unsafe
            TimeoutError: If execution times out
        """
        # Validate first
        CodeValidator.validate(code)

        # Ensure we have eager DataFrame for input
        if isinstance(df, pl.LazyFrame):
            input_df = df.collect()
        else:
            input_df = df

        # Collect extra dataframes if provided
        collected_extras: dict[str, pl.DataFrame] = {}
        if extra_dataframes:
            for name, frame in extra_dataframes.items():
                if isinstance(frame, pl.LazyFrame):
                    collected_extras[name] = frame.collect()
                else:
                    collected_extras[name] = frame

        # Prepare restricted namespace
        namespace = self._create_namespace(input_df, collected_extras)
        
        # Execute with timeout
        start_time = time.perf_counter()
        
        try:
            with timeout_context(self._timeout):
                exec(code, namespace)
        except TimeoutError:
            raise
        except Exception as e:
            # Extract line number if possible
            line_no = None
            if hasattr(e, "lineno"):
                line_no = e.lineno
            
            raise CodeExecutionError(
                f"Execution failed: {type(e).__name__}: {e}",
                code=code,
                error_line=line_no,
                original_error=e,
            )
        
        execution_time = time.perf_counter() - start_time
        
        # Extract result
        result = namespace.get("result")
        
        if result is None:
            raise CodeExecutionError(
                "Code did not produce a 'result' variable",
                code=code,
            )
        
        # Handle LazyFrame result
        if isinstance(result, pl.LazyFrame):
            try:
                result = result.collect()
            except Exception as e:
                raise CodeExecutionError(
                    f"Failed to collect LazyFrame result: {e}",
                    code=code,
                    original_error=e,
                )
        
        if not isinstance(result, pl.DataFrame):
            raise CodeExecutionError(
                f"Result is not a DataFrame, got {type(result).__name__}",
                code=code,
            )
        
        return result
    
    def _create_namespace(
        self,
        df: pl.DataFrame,
        extra_dataframes: dict[str, pl.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Create restricted execution namespace."""
        namespace = {
            # Polars
            "pl": pl,
            "polars": pl,

            # Input data (main dataframe)
            "df": df,

            # Result placeholder
            "result": None,

            # Safe builtins
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "True": True,
            "False": False,
            "None": None,
        }

        # Add extra named DataFrames for multi-table support
        # e.g., df_contatti, df_aziende for JOIN operations
        if extra_dataframes:
            for name, frame in extra_dataframes.items():
                # Sanitize name to be a valid Python identifier
                safe_name = f"df_{name.replace('-', '_').replace(' ', '_')}"
                namespace[safe_name] = frame

        return namespace

    def execute_multi_table(
        self,
        code: str,
        dataframes: dict[str, pl.DataFrame | pl.LazyFrame],
    ) -> pl.DataFrame:
        """
        Execute Polars transformation code with multiple named DataFrames.

        Args:
            code: Validated Python code
            dataframes: Dict of {table_name: DataFrame} - all tables available as df_<name>

        Returns:
            Transformed DataFrame
        """
        # Validate first
        CodeValidator.validate(code)

        # Collect all dataframes
        collected: dict[str, pl.DataFrame] = {}
        for name, frame in dataframes.items():
            if isinstance(frame, pl.LazyFrame):
                collected[name] = frame.collect()
            else:
                collected[name] = frame

        # Create namespace with all named DataFrames
        namespace = self._create_multi_table_namespace(collected)

        # Execute with timeout
        start_time = time.perf_counter()

        try:
            with timeout_context(self._timeout):
                exec(code, namespace)
        except TimeoutError:
            raise
        except Exception as e:
            line_no = getattr(e, "lineno", None)
            raise CodeExecutionError(
                f"Execution failed: {type(e).__name__}: {e}",
                code=code,
                error_line=line_no,
                original_error=e,
            )

        execution_time = time.perf_counter() - start_time

        # Extract result
        result = namespace.get("result")

        if result is None:
            raise CodeExecutionError(
                "Code did not produce a 'result' variable",
                code=code,
            )

        if isinstance(result, pl.LazyFrame):
            try:
                result = result.collect()
            except Exception as e:
                raise CodeExecutionError(
                    f"Failed to collect LazyFrame result: {e}",
                    code=code,
                    original_error=e,
                )

        if not isinstance(result, pl.DataFrame):
            raise CodeExecutionError(
                f"Result is not a DataFrame, got {type(result).__name__}",
                code=code,
            )

        return result

    def _create_multi_table_namespace(
        self,
        dataframes: dict[str, pl.DataFrame],
    ) -> dict[str, Any]:
        """Create namespace with all tables as named DataFrames (df_<name>)."""
        namespace = {
            # Polars
            "pl": pl,
            "polars": pl,

            # Result placeholder
            "result": None,

            # Safe builtins
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "True": True,
            "False": False,
            "None": None,
        }

        # Add all tables with df_ prefix
        for name, frame in dataframes.items():
            safe_name = f"df_{name.replace('-', '_').replace(' ', '_')}"
            namespace[safe_name] = frame

        # Also add first table as 'df' for backward compatibility
        if dataframes:
            first_frame = next(iter(dataframes.values()))
            namespace["df"] = first_frame

        return namespace


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def execute_transformation(
    code: str,
    df: pl.DataFrame | pl.LazyFrame,
    timeout_seconds: int = LIMITS.max_execution_seconds,
    extra_dataframes: dict[str, pl.DataFrame | pl.LazyFrame] | None = None,
) -> pl.DataFrame:
    """
    Execute transformation code on DataFrame.

    Convenience wrapper around CodeExecutor.

    Args:
        code: Validated Python code
        df: Main input DataFrame
        timeout_seconds: Max execution time
        extra_dataframes: Optional additional named DataFrames for multi-table JOIN
    """
    executor = CodeExecutor(timeout_seconds=timeout_seconds)
    return executor.execute(code, df, extra_dataframes=extra_dataframes)


def execute_multi_table_transformation(
    code: str,
    dataframes: dict[str, pl.DataFrame | pl.LazyFrame],
    timeout_seconds: int = LIMITS.max_execution_seconds,
) -> pl.DataFrame:
    """
    Execute transformation code with multiple named DataFrames.

    All tables available as df_<name> in the execution namespace.

    Args:
        code: Validated Python code
        dataframes: Dict of {table_name: DataFrame}
        timeout_seconds: Max execution time
    """
    executor = CodeExecutor(timeout_seconds=timeout_seconds)
    return executor.execute_multi_table(code, dataframes)
