"""
Error Analyzer for intelligent error recovery.

Analyzes code execution errors and provides structured context
for LLM to generate better recovery code.
"""
import re
from dataclasses import dataclass, field
from typing import Any

from polars_runner.core.constants import ExecutionErrorType


@dataclass
class ErrorAnalysis:
    """Structured analysis of a code execution error."""

    error_type: ExecutionErrorType
    message: str  # Short, clear message
    suggestion: str  # What to do to fix it
    fix_hint: str  # Keyword hint for LLM

    # Optional details
    missing_column: str | None = None
    available_columns: list[str] = field(default_factory=list)
    join_info: dict[str, str] | None = None
    code_pattern: str | None = None  # The problematic pattern found

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "fix_hint": self.fix_hint,
            "missing_column": self.missing_column,
            "available_columns": self.available_columns[:20],  # Limit for readability
            "join_info": self.join_info,
        }

    def to_recovery_context(self) -> str:
        """Format for inclusion in recovery prompt."""
        lines = [
            f"**Error Type:** {self.error_type.value}",
            f"**Problem:** {self.message}",
            f"**Solution:** {self.suggestion}",
        ]

        if self.missing_column:
            lines.append(f"**Missing Column:** `{self.missing_column}`")

        if self.join_info:
            lines.append(f"**JOIN Info:** left_on=`{self.join_info.get('left_on')}`, "
                        f"right_on=`{self.join_info.get('right_on')}`")

        if self.available_columns:
            cols_preview = ", ".join(f"`{c}`" for c in self.available_columns[:10])
            if len(self.available_columns) > 10:
                cols_preview += f" ... (+{len(self.available_columns) - 10} more)"
            lines.append(f"**Available Columns:** {cols_preview}")

        return "\n".join(lines)


class ErrorAnalyzer:
    """
    Analyzes code execution errors to provide targeted recovery suggestions.

    This enables the LLM to understand WHY the code failed and generate
    a specific fix, rather than guessing blindly.
    """

    @classmethod
    def analyze(cls, error: Exception, code: str) -> ErrorAnalysis:
        """
        Analyze an execution error and return structured context.

        Args:
            error: The exception that occurred
            code: The code that was executed

        Returns:
            ErrorAnalysis with categorized error and fix suggestions
        """
        error_str = str(error)
        error_type = type(error).__name__

        # Try each analyzer in order of specificity
        analyzers = [
            cls._analyze_join_right_key_dropped,
            cls._analyze_column_not_found,
            cls._analyze_case_sensitivity,
            cls._analyze_type_error,
            cls._analyze_pandas_syntax,
            cls._analyze_missing_result,
        ]

        for analyzer in analyzers:
            result = analyzer(error_str, code)
            if result:
                return result

        # Fallback: unknown error
        return ErrorAnalysis(
            error_type=ExecutionErrorType.UNKNOWN,
            message=f"{error_type}: {error_str[:200]}",
            suggestion="Review the error message and fix the code accordingly.",
            fix_hint="GENERAL_FIX",
        )

    @classmethod
    def _analyze_join_right_key_dropped(
        cls, error_str: str, code: str
    ) -> ErrorAnalysis | None:
        """
        Detect: ColumnNotFoundError where missing column is the RIGHT key of a JOIN.

        Polars drops the right key column when left_on != right_on.
        This is a very common error pattern.
        """
        if "ColumnNotFoundError" not in error_str and "column" not in error_str.lower():
            return None

        # Extract missing column name
        missing_col = cls._extract_missing_column(error_str)
        if not missing_col:
            return None

        # Extract JOIN info from code
        join_info = cls._extract_join_info(code)
        if not join_info:
            return None

        # Check if missing column is the right_on key
        right_on = join_info.get("right_on")
        left_on = join_info.get("left_on")

        if right_on and missing_col == right_on and left_on != right_on:
            # This is the specific pattern!
            available_cols = cls._extract_available_columns(error_str)

            return ErrorAnalysis(
                error_type=ExecutionErrorType.JOIN_RIGHT_KEY_DROPPED,
                message=f"Column '{missing_col}' was the RIGHT key of the JOIN and was dropped by Polars.",
                suggestion=(
                    f"When using JOIN with different column names (left_on='{left_on}', "
                    f"right_on='{right_on}'), Polars drops the right key. "
                    f"Use the LEFT key '{left_on}' instead, or add coalesce=False to the JOIN "
                    f"to preserve both columns."
                ),
                fix_hint="USE_LEFT_JOIN_KEY_OR_COALESCE_FALSE",
                missing_column=missing_col,
                available_columns=available_cols,
                join_info=join_info,
            )

        return None

    @classmethod
    def _analyze_column_not_found(
        cls, error_str: str, code: str
    ) -> ErrorAnalysis | None:
        """Detect: General ColumnNotFoundError (not JOIN-related)."""
        if "ColumnNotFoundError" not in error_str and "unable to find column" not in error_str.lower():
            return None

        missing_col = cls._extract_missing_column(error_str)
        available_cols = cls._extract_available_columns(error_str)

        if not missing_col:
            return None

        # Check for case sensitivity issue
        if available_cols:
            lower_missing = missing_col.lower()
            for col in available_cols:
                if col.lower() == lower_missing and col != missing_col:
                    # Found same column with different case
                    return ErrorAnalysis(
                        error_type=ExecutionErrorType.CASE_SENSITIVITY,
                        message=f"Column '{missing_col}' not found - did you mean '{col}'?",
                        suggestion=f"Column names are case-sensitive. Use '{col}' instead of '{missing_col}'.",
                        fix_hint="FIX_COLUMN_CASE",
                        missing_column=missing_col,
                        available_columns=available_cols,
                        code_pattern=f'pl.col("{missing_col}")',
                    )

        # General column not found
        return ErrorAnalysis(
            error_type=ExecutionErrorType.COLUMN_NOT_FOUND,
            message=f"Column '{missing_col}' not found in the DataFrame.",
            suggestion="Check that the column name matches exactly (case-sensitive) with the schema.",
            fix_hint="CHECK_COLUMN_NAME",
            missing_column=missing_col,
            available_columns=available_cols,
        )

    @classmethod
    def _analyze_case_sensitivity(
        cls, error_str: str, code: str
    ) -> ErrorAnalysis | None:
        """Detect: Case sensitivity issues (handled in column_not_found)."""
        # This is a fallback, main logic is in _analyze_column_not_found
        return None

    @classmethod
    def _analyze_type_error(
        cls, error_str: str, code: str
    ) -> ErrorAnalysis | None:
        """Detect: Type errors and schema mismatches."""
        if "TypeError" not in error_str and "SchemaError" not in error_str:
            return None

        # Check for string in .then() without pl.lit()
        if "str" in error_str.lower() and ".then(" in code:
            if re.search(r'\.then\s*\(\s*["\']', code):
                return ErrorAnalysis(
                    error_type=ExecutionErrorType.TYPE_MISMATCH,
                    message="String literal used in .then() without pl.lit() wrapper.",
                    suggestion="Wrap string literals with pl.lit(). Example: .then(pl.lit('value'))",
                    fix_hint="WRAP_STRING_WITH_PL_LIT",
                    code_pattern=".then('string')",
                )

        return ErrorAnalysis(
            error_type=ExecutionErrorType.TYPE_MISMATCH,
            message=f"Type error: {error_str[:150]}",
            suggestion="Use .cast() to convert column types, or wrap literals with pl.lit().",
            fix_hint="CAST_TYPES",
        )

    @classmethod
    def _analyze_pandas_syntax(
        cls, error_str: str, code: str
    ) -> ErrorAnalysis | None:
        """Detect: Pandas syntax used instead of Polars."""
        pandas_patterns = [
            (r'df\[["\']', "df['column']", "pl.col('column')"),
            (r'\.groupby\s*\(', ".groupby()", ".group_by()"),
            (r'\.iloc\s*\[', ".iloc[]", ".row() or .slice()"),
            (r'\.loc\s*\[', ".loc[]", ".filter() or .select()"),
        ]

        for pattern, wrong, correct in pandas_patterns:
            if re.search(pattern, code):
                return ErrorAnalysis(
                    error_type=ExecutionErrorType.PANDAS_SYNTAX,
                    message=f"Pandas syntax '{wrong}' detected - Polars uses different syntax.",
                    suggestion=f"Replace '{wrong}' with Polars equivalent: {correct}",
                    fix_hint="CONVERT_PANDAS_TO_POLARS",
                    code_pattern=wrong,
                )

        return None

    @classmethod
    def _analyze_missing_result(
        cls, error_str: str, code: str
    ) -> ErrorAnalysis | None:
        """Detect: Code didn't produce 'result' variable."""
        if "result" in error_str.lower() and "not" in error_str.lower():
            return ErrorAnalysis(
                error_type=ExecutionErrorType.MISSING_RESULT,
                message="Code did not produce a 'result' variable.",
                suggestion="Ensure the final DataFrame is assigned to 'result'. Example: result = df.filter(...).collect()",
                fix_hint="ASSIGN_TO_RESULT",
            )
        return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _extract_missing_column(error_str: str) -> str | None:
        """Extract the missing column name from error message."""
        # Pattern 1: "unable to find column "column_name""
        match = re.search(r'unable to find column ["\']?([^"\';\]]+)["\']?', error_str)
        if match:
            return match.group(1).strip()

        # Pattern 2: "column "column_name" not found"
        match = re.search(r'column ["\']([^"\']+)["\']', error_str, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 3: ColumnNotFoundError: column_name
        match = re.search(r'ColumnNotFoundError:\s*(\w+)', error_str)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _extract_available_columns(error_str: str) -> list[str]:
        """Extract available columns from error message."""
        # Pattern: valid columns: ["col1", "col2", ...]
        match = re.search(r'valid columns:\s*\[([^\]]+)\]', error_str)
        if match:
            cols_str = match.group(1)
            # Extract quoted column names
            cols = re.findall(r'["\']([^"\']+)["\']', cols_str)
            return cols
        return []

    @staticmethod
    def _extract_join_info(code: str) -> dict[str, str] | None:
        """Extract JOIN parameters from code."""
        # Pattern for .join(..., left_on="x", right_on="y", how="z")
        join_match = re.search(r'\.join\s*\([^)]*', code, re.DOTALL)
        if not join_match:
            return None

        join_code = join_match.group(0)

        # Extract parameters
        result = {}

        left_on = re.search(r'left_on\s*=\s*["\']([^"\']+)["\']', join_code)
        if left_on:
            result["left_on"] = left_on.group(1)

        right_on = re.search(r'right_on\s*=\s*["\']([^"\']+)["\']', join_code)
        if right_on:
            result["right_on"] = right_on.group(1)

        on_match = re.search(r'\bon\s*=\s*["\']([^"\']+)["\']', join_code)
        if on_match:
            result["on"] = on_match.group(1)

        how_match = re.search(r'how\s*=\s*["\']([^"\']+)["\']', join_code)
        if how_match:
            result["how"] = how_match.group(1)
        else:
            result["how"] = "inner"

        # Check for coalesce parameter
        coalesce_match = re.search(r'coalesce\s*=\s*(True|False)', join_code)
        if coalesce_match:
            result["coalesce"] = coalesce_match.group(1)

        return result if result else None


def has_risky_join_pattern(code: str) -> bool:
    """
    Check if code has a JOIN pattern that might cause issues.

    Risky pattern: JOIN with left_on != right_on and no coalesce=False.

    Used by RAG to filter out potentially buggy solutions.
    """
    join_info = ErrorAnalyzer._extract_join_info(code)
    if not join_info:
        return False

    left_on = join_info.get("left_on")
    right_on = join_info.get("right_on")
    coalesce = join_info.get("coalesce")

    # Risky: different column names AND no explicit coalesce=False
    if left_on and right_on and left_on != right_on:
        if coalesce != "False":
            # Check if code references right_on column after join
            # This is the actual bug pattern
            if right_on in code.split(".join")[1] if ".join" in code else False:
                return True

    return False
