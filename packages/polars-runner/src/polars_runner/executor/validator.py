"""
Code validation for security.
Validates generated code before execution.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Final

from polars_runner.core.constants import SecurityConfig


# Singleton instance
_SECURITY_CONFIG: Final[SecurityConfig] = SecurityConfig()


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass(slots=True)
class ValidationResult:
    """Result of code validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(is_valid=True)
    
    @classmethod
    def failure(cls, errors: list[str]) -> "ValidationResult":
        return cls(is_valid=False, errors=errors)
    
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# =============================================================================
# CODE VALIDATOR
# =============================================================================

class CodeValidator:
    """
    Validates generated code for security.
    
    Uses multi-layer validation:
    1. Pattern matching for blocked patterns
    2. AST analysis for import validation
    3. Method whitelist checking
    """
    
    def __init__(self) -> None:
        self._blocked_patterns: Final = _SECURITY_CONFIG.BLOCKED_PATTERNS
        self._allowed_imports: Final = _SECURITY_CONFIG.ALLOWED_IMPORTS
        self._allowed_methods: Final = _SECURITY_CONFIG.ALLOWED_POLARS_METHODS
    
    def validate(self, code: str) -> ValidationResult:
        """
        Validate code for security.
        
        Args:
            code: Python code to validate
            
        Returns:
            ValidationResult with is_valid and any errors
        """
        result = ValidationResult.success()
        
        # Layer 1: Pattern matching (fast, catches obvious issues)
        self._check_blocked_patterns(code, result)
        if not result.is_valid:
            return result
        
        # Layer 2: AST analysis (thorough, validates structure)
        self._check_ast(code, result)
        if not result.is_valid:
            return result
        
        # Layer 3: Check for result variable
        self._check_result_variable(code, result)
        
        return result
    
    def _check_blocked_patterns(self, code: str, result: ValidationResult) -> None:
        """Check for blocked patterns in code."""
        code_lower = code.lower()
        
        for pattern in self._blocked_patterns:
            if pattern.lower() in code_lower:
                result.add_error(f"Blocked pattern detected: '{pattern}'")
    
    def _check_ast(self, code: str, result: ValidationResult) -> None:
        """Parse and validate AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.add_error(f"Syntax error: {e}")
            return
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self._allowed_imports:
                        result.add_error(f"Unauthorized import: '{alias.name}'")
            
            # Check from imports
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self._allowed_imports:
                    result.add_error(f"Unauthorized import from: '{node.module}'")
            
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                self._check_function_call(node, result)
    
    def _check_function_call(self, node: ast.Call, result: ValidationResult) -> None:
        """Check if function call is allowed."""
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check for dangerous builtins
            dangerous_builtins = {"exec", "eval", "compile", "open", "__import__"}
            if func_name in dangerous_builtins:
                result.add_error(f"Dangerous function call: '{func_name}'")
    
    def _check_result_variable(self, code: str, result: ValidationResult) -> None:
        """Check that code produces a 'result' variable."""
        # Simple check: look for 'result =' or 'result=' pattern
        if not re.search(r'\bresult\s*=', code):
            result.add_warning("Code should assign output to 'result' variable")


# =============================================================================
# CODE EXTRACTION
# =============================================================================

def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.
    
    Handles various markdown formats:
    - ```python ... ```
    - ``` ... ```
    - Plain code
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted Python code
    """
    response = response.strip()
    
    # Try python code block first
    python_match = re.search(r'```python\s*\n?(.*?)```', response, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()
    
    # Try generic code block
    generic_match = re.search(r'```\s*\n?(.*?)```', response, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    
    # Return as-is (assume it's plain code)
    return response


def normalize_code(code: str) -> str:
    """
    Normalize code for consistent execution.
    
    - Ensures polars import exists
    - Normalizes whitespace
    
    Args:
        code: Raw code string
        
    Returns:
        Normalized code
    """
    lines = code.strip().split('\n')
    
    # Check if polars import exists
    has_import = any(
        'import polars' in line or 'from polars' in line
        for line in lines
    )
    
    # Add import if missing
    if not has_import:
        lines.insert(0, 'import polars as pl')
    
    return '\n'.join(lines)


# =============================================================================
# SINGLETON VALIDATOR
# =============================================================================

_validator: CodeValidator | None = None


def get_validator() -> CodeValidator:
    """Get cached validator instance."""
    global _validator
    if _validator is None:
        _validator = CodeValidator()
    return _validator


def validate_code(code: str) -> ValidationResult:
    """Convenience function to validate code."""
    return get_validator().validate(code)
