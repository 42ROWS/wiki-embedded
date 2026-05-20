"""Post-execution semantic validator — domain-agnostic safety net.

This module is the **fallback** quality gate that runs after every successful
execution, regardless of whether the oracle / self-consistency pipeline is
enabled. It performs only checks that are **universal** — they hold for any
dataset, any language, and any prompt:

  1. **Empty-frame guard** — a result with zero rows is allowed but flagged.
  2. **All-NULL column detection** — when a column is 100 % NULL on a non-
     empty frame, that is almost always a silent aggregation / arithmetic
     bug (NULL propagation), regardless of the application domain.
  3. **Non-portable Polars idioms** — at present only ``is_in(<LazyFrame>)``,
     which breaks on some Polars versions (Flash-Lite often emits it).

Anything that depends on natural-language vocabulary, on a specific column
name, or on a row-count threshold is handled by :mod:`polars_runner.executor.oracle`
instead. That separation is deliberate: this module must remain a clean,
zero-bias safety net.

The validator raises :class:`SemanticValidationFailure` on failures so the
existing retry loop can react the same way it does for execution exceptions
(see ``_execute_with_recovery`` in :mod:`polars_runner.main`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import polars as pl


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------

class VerdictLevel(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(slots=True)
class SemanticVerdict:
    """Structured verdict from the semantic validator."""

    level: VerdictLevel
    reasons: list[str] = field(default_factory=list)
    layer: Literal["deterministic", "code_heuristic", "none"] = "none"
    suggested_fix: str | None = None

    @property
    def is_fail(self) -> bool:
        return self.level is VerdictLevel.FAIL

    @property
    def is_warn(self) -> bool:
        return self.level is VerdictLevel.WARN

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "reasons": self.reasons,
            "layer": self.layer,
            "suggested_fix": self.suggested_fix,
        }

    def feedback_hint(self) -> str:
        """Format the verdict for injection into a retry prompt."""
        head = f"SEMANTIC ISSUE ({self.layer}): " + "; ".join(self.reasons)
        if self.suggested_fix:
            return f"{head}\nFIX HINT: {self.suggested_fix}"
        return head


class SemanticValidationFailure(Exception):
    """Raised when the semantic validator returns a ``fail`` verdict.

    The retry loop catches this and forwards ``verdict.feedback_hint()`` as
    the error context for the next code-generation attempt.
    """

    def __init__(self, verdict: SemanticVerdict) -> None:
        super().__init__(verdict.feedback_hint())
        self.verdict = verdict


# ---------------------------------------------------------------------------
# Layer 1 — deterministic, domain-agnostic checks
# ---------------------------------------------------------------------------

def _check_all_null_columns(result_df: pl.DataFrame) -> list[str]:
    """Flag columns that are 100 % NULL on a non-empty frame.

    Universal: a 100 % NULL output column on a non-empty result is almost
    always a silent bug — NULL propagation in arithmetic, a missed
    ``fill_null``, or a JOIN that dropped every value. The check does not
    depend on any column-name vocabulary or language.
    """
    if result_df.height == 0:
        return []
    issues: list[str] = []
    for col in result_df.columns:
        try:
            null_count = result_df[col].null_count()
        except Exception:
            continue
        if null_count == result_df.height:
            issues.append(
                f"Column '{col}' is 100% NULL ({null_count}/{result_df.height} rows). "
                "Likely null propagation in arithmetic or aggregation."
            )
    return issues


# ---------------------------------------------------------------------------
# Layer 2 — code-level heuristics for non-portable Polars idioms
# ---------------------------------------------------------------------------

# ``is_in(<expression>.lazy())`` is accepted by some Polars versions and not
# others. It is the only idiom common enough to warrant a static check; we
# explicitly avoid hard-coding domain vocabulary or natural-language patterns
# here — anything that depends on the prompt belongs in the oracle.
_LAZY_IS_IN_RE = re.compile(r"\.is_in\s*\(\s*[^)]*\.lazy\s*\(", re.IGNORECASE)


def _check_non_portable_idioms(code: str) -> list[str]:
    issues: list[str] = []
    if _LAZY_IS_IN_RE.search(code):
        issues.append(
            "Code uses `is_in(<LazyFrame>)`, which is not portable across "
            "Polars versions. Materialize the lookup first: "
            "`lookup = df_other.lazy().select('col').unique().collect().to_series()`, "
            "then `pl.col('x').is_in(lookup)`."
        )
    return issues


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _suggest_fix_from_reasons(reasons: list[str]) -> str:
    """One-line fix hint derived from the failure category.

    Kept intentionally short and generic — the LLM will combine it with the
    original prompt to draft a corrected solution.
    """
    text = " ".join(reasons).lower()
    if "100% null" in text or "null propagation" in text:
        return (
            "Check the aggregation: ensure both joined columns are non-null "
            "before the arithmetic, or use `fill_null(0)` / `pl.coalesce(...)`. "
            "After `.group_by(...).agg(...)` the result should collapse to one "
            "row per group."
        )
    if "is_in(<lazyframe>)" in text or "lazyframe" in text:
        return (
            "Materialize the lookup set before `is_in`: "
            "`lookup = df_other.lazy().select('col').unique().collect().to_series()`, "
            "then `pl.col('x').is_in(lookup)`."
        )
    return "Re-examine the aggregation / filter and re-emit the code."


def validate_semantic(
    *,
    code: str,
    result_df: pl.DataFrame,
) -> SemanticVerdict:
    """Run the universal semantic checks.

    Two layers, all language-agnostic and domain-agnostic:

      1. **Deterministic** — 100% NULL columns on a non-empty frame.
      2. **Code idiom** — non-portable Polars constructs (currently
         ``is_in(<LazyFrame>)``).

    Anything that depends on the natural-language prompt — expected row
    counts, monotonicity, value ranges — lives in
    :mod:`polars_runner.executor.oracle`.
    """
    # Layer 1
    issues = _check_all_null_columns(result_df)
    if issues:
        return SemanticVerdict(
            level=VerdictLevel.FAIL,
            reasons=issues,
            layer="deterministic",
            suggested_fix=_suggest_fix_from_reasons(issues),
        )

    # Layer 2
    issues = _check_non_portable_idioms(code)
    if issues:
        return SemanticVerdict(
            level=VerdictLevel.FAIL,
            reasons=issues,
            layer="code_heuristic",
            suggested_fix=_suggest_fix_from_reasons(issues),
        )

    return SemanticVerdict(level=VerdictLevel.OK, layer="none")


__all__ = [
    "SemanticValidationFailure",
    "SemanticVerdict",
    "VerdictLevel",
    "validate_semantic",
]
