"""Property-Based Testing oracle for LLM-generated data transformations.

The oracle is a structured set of *universal* properties extracted from the
user's natural-language prompt. Any valid output of the transformation must
satisfy all properties. The oracle is dataset-agnostic and language-agnostic:
properties are derived from the prompt's intent (via a premium LLM), not from
hard-coded heuristics or golden-answer tables.

Pattern references:
    * Vikram et al., "Can Large Language Models Write Good Property-Based
      Tests?" (arxiv 2307.04346).
    * "From Prompts to Properties: Rethinking LLM Code Generation with
      Property-Based Testing" (FSE 2024, dl.acm.org/10.1145/3696630.3728702).
    * "Use Property-Based Testing to Bridge LLM Code Generation and
      Validation" (arxiv 2506.18315).
    * Zhang et al., "ALGO: Synthesizing Algorithmic Programs with LLM-
      Generated Oracle Verifiers" (ICLR 2024).

Pipeline summary:
    1. ``generate_oracle(prompt, schema, llm)`` → premium LLM produces an
       :class:`Oracle` (one call per user prompt).
    2. ``verify_against_oracle(oracle, df)`` → pure Python checker, no LLM in
       the loop. Returns a structured :class:`OracleVerdict`.

The verifier is deliberately conservative: every check tolerates the absence
of evidence (no constraint declared → no failure), so an empty oracle yields
a passing verdict. This keeps the oracle non-blocking when the premium LLM
cannot extract meaningful properties.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ValueConstraint:
    """Numeric / categorical constraint on a single output column.

    All fields are optional: an empty constraint is a no-op. Mixing
    ``min/max`` with ``allowed_values`` is supported (both checked).
    """

    min: float | None = None
    max: float | None = None
    allowed_values: tuple[Any, ...] | None = None
    must_be_unique: bool = False


@dataclass(frozen=True, slots=True)
class Oracle:
    """Universal output contract extracted from a user prompt.

    Each field is independently optional. The verifier interprets unset
    fields as "no claim" and never fails on missing information.
    """

    # Expected output cardinality. Either bound can be ``None``.
    expected_rows_range: tuple[int | None, int | None] = (None, None)

    # Column-level expectations.
    required_columns: frozenset[str] = frozenset()
    non_null_columns: frozenset[str] = frozenset()
    value_constraints: dict[str, ValueConstraint] = field(default_factory=dict)

    # If the prompt asks for a sorted output, declare the direction here.
    monotonicity: dict[str, Literal["asc", "desc"]] = field(default_factory=dict)

    # Provenance — useful for debugging and for stamping vectors in the
    # learned-skill store.
    source_prompt: str = ""
    source_model: str = ""
    raw_response: str = ""

    @property
    def is_empty(self) -> bool:
        """An oracle with no constraints is treated as a no-op by the verifier."""
        return (
            self.expected_rows_range == (None, None)
            and not self.required_columns
            and not self.non_null_columns
            and not self.value_constraints
            and not self.monotonicity
        )


@dataclass(frozen=True, slots=True)
class OracleVerdict:
    """Outcome of evaluating an :class:`Oracle` against a :class:`polars.DataFrame`."""

    passed: bool
    failed_rules: tuple[str, ...] = ()
    score: float = 100.0  # 0-100, percentage of declared rules satisfied
    feedback_hint: str = ""  # human-readable summary suitable for retry prompts

    @classmethod
    def passing(cls) -> "OracleVerdict":
        return cls(passed=True, failed_rules=(), score=100.0, feedback_hint="")


class LLMClient(Protocol):
    """Minimal contract the oracle generator needs from an LLM client.

    Implementations are expected to honour ``temperature`` and to return
    raw text. JSON parsing happens here, not in the client.
    """

    def complete(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_mime_type: str | None = None,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Generation — premium LLM call, one per user prompt
# ---------------------------------------------------------------------------

_ORACLE_INSTRUCTION: Final[str] = """\
You are a careful data-analysis reviewer. Given a USER QUESTION about one or
more Polars dataframes, plus the EXACT INPUT SCHEMA, produce a strict and
MINIMAL contract that any valid result frame must satisfy. The contract is
fed to a deterministic Python verifier — do not invent constraints the
prompt does not imply, and use ONLY the column names that appear verbatim in
the input schema (no synonyms, no abbreviations, no invented columns).

Return ONE JSON object with this schema. Omit any field you cannot infer
*from the prompt itself*; the verifier treats absence as "no claim".

{{
  "expected_rows_range": [min:int|null, max:int|null],
    // Inferred only from the prompt. Examples:
    //   "top 10 ..."           -> [1, 10]
    //   "for each <category>"  -> [1, n_distinct(<category column in the input>)]
    //   no row hint            -> [null, null]
  "required_columns":  ["col1", "col2", ...],
    // Column names that MUST appear in the result frame. Use names verbatim
    // from the input schema. If the prompt asks for an aggregation that
    // produces a new column (e.g. "count of X"), name it with the standard
    // Polars convention ("count", "len", "<agg>_<source_col>" — never invent
    // a new domain word).
  "non_null_columns":  ["col1", ...],
    // Columns that must have zero NULLs in the result. Use for arithmetic
    // and aggregation outputs that should not silently propagate NULL.
  "value_constraints": {{
    "col": {{
      "min": number|null, "max": number|null,
      "allowed_values": [list|null],
      "must_be_unique": bool
    }}
  }},
  "monotonicity": {{ "col": "asc"|"desc" }}
    // Only if the prompt explicitly asks for a sorted answer.
}}

Guidelines:
- Be language-agnostic: the prompt may be in any language.
- Be domain-agnostic: do not assume vocabulary from any particular dataset.
- Multi-table input: the schema below is split into ``# df_<name>`` sections.
  At runtime the Polars code references them as ``df_<name>``. The result
  frame inherits column names from whichever table(s) the prompt joins,
  unless the prompt explicitly renames them.
- Prefer omission to invention. An empty contract is acceptable.
- Output ONLY the JSON object. No prose, no code fence.

INPUT SCHEMA:
{schema_block}

USER QUESTION:
{prompt}
"""


def _format_single_schema(schema: Any) -> list[str]:
    """Render a single schema as ``col: dtype`` lines (no header)."""
    if isinstance(schema, pl.DataFrame):
        return [f"  {c}: {schema.schema[c]}" for c in schema.columns]
    cols = getattr(schema, "columns", None)
    if isinstance(cols, dict):
        lines: list[str] = []
        for name, info in cols.items():
            dtype = getattr(info, "dtype", None) or getattr(info, "type", None) or info
            lines.append(f"  {name}: {dtype}")
        return lines
    if isinstance(schema, dict):
        return [f"  {k}: {v}" for k, v in schema.items()]
    return [str(schema)]


def _format_schema(schema: Any) -> str:
    """Render the dataset schema as input for the oracle prompt.

    Supports three call patterns so the oracle sees the *real* column names
    a Polars transformation will use:

    1. **Multi-table** — ``dict[table_name, SchemaInfo | dict | DataFrame]``.
       Rendered as one section per table, prefixed with ``# df_<name>``.
       The dataframes are referenced in code as ``df_<name>``, so the oracle
       can reason about JOIN keys and per-table column ownership.
    2. **Single table** — anything else (``SchemaInfo``, ``dict``, ``DataFrame``).
    3. **Unknown** — falls back to ``str(schema)``.

    Distinguishing (1) from (2): a multi-table payload is a plain ``dict``
    whose values are themselves schema-like (``SchemaInfo`` instance,
    ``DataFrame``, or nested ``dict`` with non-string-leaf values). A plain
    ``dict[str, str]`` is treated as a single schema map (case 2).
    """
    if isinstance(schema, dict) and schema and not all(
        isinstance(v, (str, int, float)) for v in schema.values()
    ):
        sections: list[str] = []
        for table_name, table_schema in schema.items():
            lines = _format_single_schema(table_schema)
            sections.append(f"# df_{table_name}\n" + "\n".join(lines))
        return "\n\n".join(sections)
    return "\n".join(_format_single_schema(schema))


def _coerce_value_constraint(raw: Any) -> ValueConstraint:
    if not isinstance(raw, dict):
        return ValueConstraint()
    allowed = raw.get("allowed_values")
    if allowed is not None and not isinstance(allowed, (list, tuple)):
        allowed = None
    return ValueConstraint(
        min=raw.get("min") if isinstance(raw.get("min"), (int, float)) else None,
        max=raw.get("max") if isinstance(raw.get("max"), (int, float)) else None,
        allowed_values=tuple(allowed) if allowed is not None else None,
        must_be_unique=bool(raw.get("must_be_unique", False)),
    )


def _coerce_rows_range(raw: Any) -> tuple[int | None, int | None]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = raw
        lo_i = int(lo) if isinstance(lo, (int, float)) else None
        hi_i = int(hi) if isinstance(hi, (int, float)) else None
        return (lo_i, hi_i)
    return (None, None)


def _parse_oracle_response(raw_text: str, prompt: str, model: str) -> Oracle:
    """Parse the premium LLM's JSON response into a typed :class:`Oracle`.

    Robust to surrounding whitespace, stray code fences, and partial fields.
    Any unparseable response yields an empty oracle (no-op verifier).
    """
    text = raw_text.strip()
    # Strip optional ```json fences defensively (the prompt asks not to use
    # them, but Gemini occasionally does anyway).
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3].rstrip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Oracle JSON parse failed; defaulting to empty oracle. raw=%r", raw_text[:200])
        return Oracle(source_prompt=prompt, source_model=model, raw_response=raw_text)

    if not isinstance(parsed, dict):
        return Oracle(source_prompt=prompt, source_model=model, raw_response=raw_text)

    required = parsed.get("required_columns") or []
    non_null = parsed.get("non_null_columns") or []
    value_cs = parsed.get("value_constraints") or {}
    monotone = parsed.get("monotonicity") or {}

    return Oracle(
        expected_rows_range=_coerce_rows_range(parsed.get("expected_rows_range")),
        required_columns=frozenset(str(c) for c in required if isinstance(c, str)),
        non_null_columns=frozenset(str(c) for c in non_null if isinstance(c, str)),
        value_constraints={
            str(col): _coerce_value_constraint(vc)
            for col, vc in value_cs.items()
            if isinstance(col, str)
        },
        monotonicity={
            str(col): direction
            for col, direction in monotone.items()
            if isinstance(col, str) and direction in ("asc", "desc")
        },
        source_prompt=prompt,
        source_model=model,
        raw_response=raw_text,
    )


def generate_oracle(
    *,
    prompt: str,
    schema: Any,
    llm: LLMClient,
    model: str = "gemini-2.5-pro",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Oracle:
    """Ask a premium LLM to extract a property contract from the user prompt.

    A single LLM call. Returns an empty oracle on any failure — never raises.
    The verifier interprets an empty oracle as "no claim", so a downstream
    pipeline that uses the oracle as a gate stays open when extraction fails.
    """
    instruction = _ORACLE_INSTRUCTION.format(
        schema_block=_format_schema(schema),
        prompt=prompt,
    )
    try:
        raw = llm.complete(
            prompt=instruction,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_mime_type="application/json",
        )
    except Exception as e:
        logger.warning("Oracle generation failed (%s); using empty oracle.", e)
        return Oracle(source_prompt=prompt, source_model=model)

    oracle = _parse_oracle_response(raw, prompt=prompt, model=model)
    if oracle.is_empty:
        logger.info("Oracle is empty for prompt %r — verifier will be a no-op.", prompt[:80])
    return oracle


# ---------------------------------------------------------------------------
# Verification — pure Python, no LLM in the loop
# ---------------------------------------------------------------------------

def _check_rows_range(oracle: Oracle, df: pl.DataFrame) -> str | None:
    lo, hi = oracle.expected_rows_range
    if lo is not None and df.height < lo:
        return f"expected_rows_range: got {df.height} rows, minimum is {lo}"
    if hi is not None and df.height > hi:
        return f"expected_rows_range: got {df.height} rows, maximum is {hi}"
    return None


def _check_required_columns(oracle: Oracle, df: pl.DataFrame) -> str | None:
    if not oracle.required_columns:
        return None
    have = {c.lower() for c in df.columns}
    missing = sorted(c for c in oracle.required_columns if c.lower() not in have)
    if missing:
        return f"required_columns: missing {missing}"
    return None


def _check_non_null(oracle: Oracle, df: pl.DataFrame) -> list[str]:
    issues: list[str] = []
    columns_lc = {c.lower(): c for c in df.columns}
    for col in oracle.non_null_columns:
        actual = columns_lc.get(col.lower())
        if actual is None:
            continue  # missing column already reported by required_columns
        try:
            nulls = df[actual].null_count()
        except Exception:
            continue
        if nulls > 0:
            issues.append(
                f"non_null_columns: '{actual}' has {nulls}/{df.height} NULLs "
                "(likely null propagation in aggregation)"
            )
    return issues


def _check_value_constraints(oracle: Oracle, df: pl.DataFrame) -> list[str]:
    issues: list[str] = []
    columns_lc = {c.lower(): c for c in df.columns}
    for col, vc in oracle.value_constraints.items():
        actual = columns_lc.get(col.lower())
        if actual is None or df.height == 0:
            continue
        series = df[actual]
        try:
            if vc.min is not None:
                lo = series.min()
                if lo is not None and lo < vc.min:
                    issues.append(f"value_constraints['{actual}'].min: got {lo}, expected ≥ {vc.min}")
            if vc.max is not None:
                hi = series.max()
                if hi is not None and hi > vc.max:
                    issues.append(f"value_constraints['{actual}'].max: got {hi}, expected ≤ {vc.max}")
            if vc.allowed_values is not None:
                bad = (
                    series.is_in(list(vc.allowed_values)).not_().sum()
                    if vc.allowed_values
                    else 0
                )
                if bad and bad > 0:
                    issues.append(
                        f"value_constraints['{actual}'].allowed_values: "
                        f"{bad} rows outside the allowed set"
                    )
            if vc.must_be_unique and series.n_unique() != df.height:
                issues.append(
                    f"value_constraints['{actual}'].must_be_unique: "
                    f"got {series.n_unique()} distinct over {df.height} rows"
                )
        except Exception as e:
            logger.debug("Skipped value_constraints['%s'] due to: %s", actual, e)
    return issues


def _check_monotonicity(oracle: Oracle, df: pl.DataFrame) -> list[str]:
    issues: list[str] = []
    if df.height < 2:
        return issues
    columns_lc = {c.lower(): c for c in df.columns}
    for col, direction in oracle.monotonicity.items():
        actual = columns_lc.get(col.lower())
        if actual is None:
            continue
        try:
            sorted_df = df.sort(actual, descending=(direction == "desc"))
            if not df[actual].equals(sorted_df[actual]):
                issues.append(f"monotonicity['{actual}']: not sorted {direction}")
        except Exception as e:
            logger.debug("Skipped monotonicity['%s'] due to: %s", actual, e)
    return issues


def verify_against_oracle(oracle: Oracle, df: pl.DataFrame) -> OracleVerdict:
    """Evaluate the oracle against a result DataFrame.

    Pure Python, deterministic, no network calls. An empty oracle always
    passes — this is the design (the verifier never invents constraints).
    """
    if oracle.is_empty:
        return OracleVerdict.passing()

    failed: list[str] = []
    if (msg := _check_rows_range(oracle, df)) is not None:
        failed.append(msg)
    if (msg := _check_required_columns(oracle, df)) is not None:
        failed.append(msg)
    failed.extend(_check_non_null(oracle, df))
    failed.extend(_check_value_constraints(oracle, df))
    failed.extend(_check_monotonicity(oracle, df))

    # Score = fraction of declared rules that passed. Each kind of rule
    # contributes once (a binary pass / fail per category).
    declared_rule_kinds = sum(
        1
        for present in (
            oracle.expected_rows_range != (None, None),
            bool(oracle.required_columns),
            bool(oracle.non_null_columns),
            bool(oracle.value_constraints),
            bool(oracle.monotonicity),
        )
        if present
    )
    failed_kinds = len({rule.split(":", 1)[0] for rule in failed})
    score = (
        100.0 * (declared_rule_kinds - failed_kinds) / declared_rule_kinds
        if declared_rule_kinds
        else 100.0
    )

    feedback = ""
    if failed:
        feedback = "Oracle failed on: " + "; ".join(failed[:5])
        if len(failed) > 5:
            feedback += f" (+{len(failed) - 5} more)"

    return OracleVerdict(
        passed=not failed,
        failed_rules=tuple(failed),
        score=round(score, 2),
        feedback_hint=feedback,
    )


__all__ = [
    "LLMClient",
    "Oracle",
    "OracleVerdict",
    "ValueConstraint",
    "generate_oracle",
    "verify_against_oracle",
]
