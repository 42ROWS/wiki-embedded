"""
Optimized prompts for Polars code generation.
Designed to minimize Pandas syntax hallucination.
"""

# =============================================================================
# SYSTEM PROMPT - Core instructions for LLM
# =============================================================================

POLARS_SYSTEM_PROMPT = """You are an expert Polars data transformation assistant.
Your task is to generate ONLY valid Polars code - never Pandas syntax.

## CRITICAL RULES - FOLLOW EXACTLY

### ⚠️ COLUMN NAMES ARE CASE-SENSITIVE - MOST IMPORTANT RULE
Column names must match EXACTLY as shown in the schema, including capitalization!
- If schema shows `Country`, use `pl.col("Country")` NOT `pl.col("country")`
- If schema shows `First Name`, use `pl.col("First Name")` NOT `pl.col("first_name")`
- If schema shows `Category`, use `pl.col("Category")` NOT `pl.col("category")`
- ALWAYS copy column names exactly from the schema provided

### FORBIDDEN - NEVER USE THESE (Common Mistakes)
❌ `.to_frame()` - Does NOT exist on LazyFrame
❌ `.to_pandas()` on LazyFrame - Must `.collect()` first
❌ `groupby()` - Use `group_by()` instead
❌ `distinct()` - Use `unique()` instead  
❌ `order_by()` - Use `sort()` instead
❌ `df["column"]` - Use `pl.col("column")` instead
❌ `df[df.x > 5]` - Use `.filter(pl.col("x") > 5)` instead
❌ `.alias()` on LazyFrame - Use `.alias()` only on column expressions inside `.select()` or `.with_columns()`
❌ `df.alias("name")` - WRONG! Use `pl.col("x").alias("name")` inside expressions

### Syntax Rules
1. Use `group_by()` NOT `groupby()`
2. Use `pl.col("name")` for column references, NOT `df["name"]`
3. Use `.filter()` with expressions, NOT boolean indexing like `df[df.x > 5]`
4. Use `.select()` and `.with_columns()` for column operations
5. Aggregations go inside `.agg()` after `.group_by()`
6. Use `.alias()` to name computed columns
7. Use `unique()` NOT `distinct()` - Polars uses `unique()` for deduplication
8. Use `head(n)` or `limit(n)` for top N rows
9. Use `sort()` NOT `order_by()`
10. To count rows: use `pl.len()` inside `.agg()` or `.select(pl.len())`

### Performance Rules
1. ALWAYS use lazy evaluation: start with `.lazy()`, end with `.collect()`
2. Chain operations in a single expression when possible
3. Filter early to reduce data size
4. Select only needed columns early

### Code Structure
- Import polars as `pl` at the top
- Input DataFrame is available as `df`
- Store final result in variable `result`
- Return a DataFrame, not a Series

## CORRECT PATTERNS

### Filtering
```python
# CORRECT
df.filter(pl.col("price") > 100)
df.filter((pl.col("status") == "active") & (pl.col("value") > 0))

# WRONG - Pandas style
df[df["price"] > 100]  # NEVER use this
```

### Selecting Columns
```python
# CORRECT
df.select("col1", "col2", "col3")
df.select(pl.col("col1"), pl.col("col2").alias("renamed"))

# WRONG
df[["col1", "col2"]]  # NEVER use this
```

### Adding/Modifying Columns
```python
# CORRECT
df.with_columns(
    (pl.col("price") * pl.col("quantity")).alias("total"),
    pl.col("name").str.to_uppercase().alias("name_upper"),
)

# WRONG
df["total"] = df["price"] * df["quantity"]  # NEVER use this
```

### Grouping and Aggregating
```python
# CORRECT
df.group_by("category").agg(
    pl.col("sales").sum().alias("total_sales"),
    pl.col("price").mean().alias("avg_price"),
    pl.len().alias("count"),
)

# WRONG
df.groupby("category")["sales"].sum()  # NEVER use groupby
```

### Sorting
```python
# CORRECT
df.sort("date", descending=True)
df.sort("col1", "col2", descending=[True, False])
```

### Conditionals
```python
# CORRECT
df.with_columns(
    pl.when(pl.col("value") > 100)
    .then(pl.lit("high"))
    .when(pl.col("value") > 50)
    .then(pl.lit("medium"))
    .otherwise(pl.lit("low"))
    .alias("tier")
)
```

### Window Functions
```python
# CORRECT
df.with_columns(
    pl.col("sales").sum().over("region").alias("region_total"),
    pl.col("value").rank().over("category").alias("rank_in_category"),
)
```

### Counting Rows
```python
# CORRECT - Count total rows
result = df.select(pl.len().alias("total_count"))

# CORRECT - Count rows in a group
result = df.group_by("category").agg(pl.len().alias("count"))

# CORRECT - Count unique/distinct values in a column
result = df.select(pl.col("country").n_unique().alias("unique_count"))

# WRONG - These don't exist or are incorrect
df.count()  # NEVER use this
len(df)  # Don't use Python len() for counting
df.n_unique()  # WRONG - n_unique() is on columns, not DataFrames
df.alias("name")  # WRONG - alias() is for column expressions only
```

### Full Lazy Pipeline (PREFERRED)
```python
import polars as pl

result = (
    df.lazy()
    .filter(pl.col("status") == "active")
    .select("id", "category", "value")
    .group_by("category")
    .agg(
        pl.col("value").sum().alias("total"),
        pl.col("value").mean().alias("average"),
        pl.len().alias("count"),
    )
    .sort("total", descending=True)
    .collect()
)
```

### Multi-Table JOIN (when multiple tables are provided)
When you receive multiple named DataFrames (e.g., df_contacts, df_companies), use Polars JOIN:
```python
import polars as pl

# Example: Join contacts with companies on company_id
# IMPORTANT: Use coalesce=False when left_on != right_on to keep both columns!
result = (
    df_contacts.lazy()
    .join(
        df_companies.lazy(),
        left_on="company_id",
        right_on="id",
        how="left",
        coalesce=False  # REQUIRED when column names differ!
    )
    .select(
        pl.col("name"),           # from contacts
        pl.col("company_name"),   # from companies
        pl.col("email"),
    )
    .collect()
)
```

Join types:
- `how="inner"`: Only matching rows from both tables
- `how="left"`: All rows from left table, matching from right
- `how="outer"`: All rows from both tables
- `how="cross"`: Cartesian product (every combination)

### ⚠️ CRITICAL: JOIN Column Behavior with Different Names
When joining on columns with DIFFERENT names (left_on != right_on), Polars DROPS the right join key column BY DEFAULT!

**Solution 1 - Use coalesce=False (BEST/SAFEST):**
```python
# coalesce=False keeps BOTH join columns - no information loss!
result = (
    df_contatti.lazy()
    .join(
        df_aziende.lazy(),
        left_on="company_name",
        right_on="ragione_sociale",
        how="left",
        coalesce=False  # This preserves BOTH columns!
    )
    .select(
        pl.col("company_name"),      # Left key - always available
        pl.col("ragione_sociale"),   # Right key - NOW available thanks to coalesce=False!
        pl.col("nome"),
        pl.col("cognome"),
    )
    .collect()
)
```

**Solution 2 - Use only the LEFT column name (when you don't need the right key):**
```python
# Without coalesce=False, only LEFT key remains after JOIN
result = (
    df_contatti.lazy()
    .join(df_aziende.lazy(), left_on="company_name", right_on="ragione_sociale", how="left")
    .select(
        pl.col("company_name"),  # Use LEFT key, "ragione_sociale" doesn't exist!
        pl.col("nome"),
        pl.col("cognome"),
    )
    .collect()
)
```

**Solution 3 - Rename before join (when you need same column name):**
```python
result = (
    df_contatti.lazy()
    .join(
        df_aziende.lazy().rename({"ragione_sociale": "company_name"}),
        on="company_name",  # Now same name!
        how="left"
    )
    .collect()
)
```

REMEMBER: When left_on != right_on, ALWAYS use coalesce=False OR use only the LEFT column name after JOIN!

## OUTPUT FORMAT

Return ONLY Python code. No markdown, no explanations, no comments.
The code must be directly executable.
"""


# =============================================================================
# USER PROMPT TEMPLATE
# =============================================================================

USER_PROMPT_TEMPLATE = """## Dataset Schema
{schema}

## Total Rows
{row_count:,}

## User Request
{user_prompt}

## IMPORTANT REMINDER
- Column names are CASE-SENSITIVE! Use exact names from the schema above.
- If the schema shows "Country", use "Country" not "country"
- If the schema shows "Category", use "Category" not "category"

## Instructions
Generate Polars code to perform this transformation.
- Input DataFrame is `df`
- Store result in `result`
- Use lazy evaluation for performance
- Return a DataFrame

Generate the code now:"""


# Multi-table template (for JOIN operations)
MULTI_TABLE_PROMPT_TEMPLATE = """## Multiple Tables Available

{tables_section}

## User Request
{user_prompt}

## IMPORTANT REMINDER
- Column names are CASE-SENSITIVE! Use exact names from each table's schema.
- Each table is available as a separate DataFrame: {df_names}
- Use `.join()` to combine tables based on common keys.

## Instructions
Generate Polars code to perform this transformation.
- Input DataFrames: {df_names}
- Store result in `result`
- Use lazy evaluation for performance
- Return a DataFrame

Generate the code now:"""


# =============================================================================
# ERROR RECOVERY PROMPT
# =============================================================================

# Legacy template (kept for backward compatibility)
ERROR_RECOVERY_TEMPLATE = """## Previous Attempt Failed

The previous code generated an error:
```
{error_message}
```

Previous code:
```python
{previous_code}
```

## Dataset Schema (unchanged)
{schema}

## Original Request
{user_prompt}

## Instructions
Fix the code to handle this error. Common fixes:
- ColumnNotFoundError: Check column names match schema EXACTLY including capitalization!
  - If schema shows "Country", use "Country" NOT "country"
  - If schema shows "Category", use "Category" NOT "category"
  - **JOIN issue**: If the missing column is the RIGHT join key (e.g., "ragione_sociale" when joining left_on="company_name", right_on="ragione_sociale"), Polars DROPS it! Use coalesce=False in the JOIN, or use the LEFT key name instead.
- SchemaError: Use proper type casting with `.cast()`
- String in .then(): Wrap with `pl.lit()` not bare string

Generate corrected code now:"""


# =============================================================================
# INTELLIGENT ERROR RECOVERY PROMPT (V2)
# =============================================================================

ERROR_RECOVERY_TEMPLATE_V2 = """## ⚠️ EXECUTION ERROR - INTELLIGENT RECOVERY

### Error Analysis
{error_analysis}

### Technical Error (for reference)
```
{error_message_short}
```

### Failed Code
```python
{previous_code}
```

{schema_section}

## Original Request
{user_prompt}

## 🔧 SPECIFIC FIX REQUIRED
{fix_instructions}

Generate corrected code now:"""


# Fix instructions per error type
ERROR_FIX_INSTRUCTIONS = {
    "join_right_key_dropped": """
**THE PROBLEM**: You used a JOIN with different column names (left_on != right_on), and Polars dropped the right key column.

**THE FIX**: Add `coalesce=False` to the JOIN to preserve both columns:
```python
.join(
    df_other.lazy(),
    left_on="your_left_column",
    right_on="your_right_column",
    how="left",
    coalesce=False  # ADD THIS!
)
```

Alternatively, use only the LEFT column name after the join (the right key no longer exists).
""",

    "column_not_found": """
**THE PROBLEM**: A column name doesn't exist in the DataFrame.

**THE FIX**:
1. Check the column name matches EXACTLY (case-sensitive!) with the schema
2. If it was a JOIN right key, add coalesce=False or use the left key name
3. Verify you're referencing the correct table's columns
""",

    "case_sensitivity": """
**THE PROBLEM**: Column name case doesn't match the schema.

**THE FIX**: Use the EXACT column name from the schema, including capitalization.
Example: If schema shows "Country", use `pl.col("Country")` NOT `pl.col("country")`
""",

    "type_mismatch": """
**THE PROBLEM**: Type error - possibly a string literal without pl.lit() or type mismatch.

**THE FIX**:
1. Wrap string literals with pl.lit(): `.then(pl.lit("value"))` NOT `.then("value")`
2. Use .cast() for type conversions: `.cast(pl.Int64)`, `.cast(pl.Utf8)`
""",

    "pandas_syntax": """
**THE PROBLEM**: Pandas syntax was used instead of Polars.

**THE FIX**: Convert to Polars syntax:
- `df["col"]` → `pl.col("col")`
- `.groupby()` → `.group_by()`
- `df[df.x > 5]` → `df.filter(pl.col("x") > 5)`
- `.iloc[]` → `.row()` or `.slice()`
""",

    "missing_result": """
**THE PROBLEM**: Code didn't assign the final DataFrame to 'result'.

**THE FIX**: Ensure the last line assigns to `result`:
```python
result = (
    df.lazy()
    .filter(...)
    .collect()
)
```
""",

    "unknown": """
**THE PROBLEM**: An unexpected error occurred.

**THE FIX**: Review the error message carefully and:
1. Check all column names match the schema exactly
2. Ensure proper Polars syntax (not Pandas)
3. Verify type compatibility in operations
4. Add coalesce=False to any JOIN with different column names
""",
}


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "request": "Count total rows/customers",
        "code": """import polars as pl

result = (
    df.lazy()
    .select(pl.len().alias("total_count"))
    .collect()
)""",
    },
    {
        "request": "Count unique/distinct countries (schema has 'Country' column)",
        "code": """import polars as pl

result = (
    df.lazy()
    .select(pl.col("Country").n_unique().alias("unique_countries"))
    .collect()
)""",
    },
    {
        "request": "Show top 10 rows sorted by name",
        "code": """import polars as pl

result = (
    df.lazy()
    .sort("name")
    .head(10)
    .collect()
)""",
    },
    {
        "request": "Group by region and calculate total sales",
        "code": """import polars as pl

result = (
    df.lazy()
    .group_by("region")
    .agg(pl.col("sales").sum().alias("total_sales"))
    .sort("total_sales", descending=True)
    .collect()
)""",
    },
    {
        "request": "Filter rows where status is active and value > 100, then select name and value columns",
        "code": """import polars as pl

result = (
    df.lazy()
    .filter((pl.col("status") == "active") & (pl.col("value") > 100))
    .select("name", "value")
    .collect()
)""",
    },
    {
        "request": "Add a column with category tier based on value ranges",
        "code": """import polars as pl

result = (
    df.lazy()
    .with_columns(
        pl.when(pl.col("value") > 1000)
        .then(pl.lit("premium"))
        .when(pl.col("value") > 500)
        .then(pl.lit("standard"))
        .otherwise(pl.lit("basic"))
        .alias("tier")
    )
    .collect()
)""",
    },
    {
        "request": "Calculate running total of sales per customer",
        "code": """import polars as pl

result = (
    df.lazy()
    .sort("date")
    .with_columns(
        pl.col("sales").cum_sum().over("customer_id").alias("running_total")
    )
    .collect()
)""",
    },
    {
        "request": "Group by Country and count customers (schema has 'Country' with capital C)",
        "code": """import polars as pl

result = (
    df.lazy()
    .group_by("Country")
    .agg(pl.len().alias("customer_count"))
    .sort("customer_count", descending=True)
    .collect()
)""",
    },
    {
        "request": "Join contacts (df_contatti) with companies (df_aziende) using company_name=ragione_sociale, show company name, contact name",
        "code": """import polars as pl

# BEST PRACTICE: Use coalesce=False when joining on different column names
# This preserves BOTH join key columns (company_name AND ragione_sociale)
result = (
    df_contatti.lazy()
    .join(
        df_aziende.lazy(),
        left_on="company_name",
        right_on="ragione_sociale",
        how="left",
        coalesce=False  # CRITICAL: preserves both join key columns!
    )
    .select(
        pl.col("company_name"),
        pl.col("ragione_sociale"),  # Available because coalesce=False
        pl.col("nome"),
        pl.col("cognome"),
    )
    .collect()
)""",
    },
]


def build_few_shot_section() -> str:
    """Build few-shot examples section."""
    lines = ["## Examples\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"### Example {i}")
        lines.append(f"Request: {ex['request']}")
        lines.append(f"```python\n{ex['code']}\n```\n")
    return "\n".join(lines)


def build_rag_context_section(similar_codes: list) -> str:
    """Build RAG context section from similar transformations."""
    if not similar_codes:
        return ""
    
    lines = ["## 🧠 Similar Transformations Found in Memory\n"]
    lines.append(
        "The system has found similar transformations from past successful runs. "
        "You can learn from these patterns and adapt them to the current request.\n"
    )
    
    for i, sim in enumerate(similar_codes, 1):
        lines.append(f"### Memory {i} (Similarity: {sim.score:.1%}, Quality: {sim.quality_score:.0f}/100)")
        lines.append(f"**Original Request:** {sim.prompt}")
        lines.append(f"**Successful Code:**")
        lines.append(f"```python\n{sim.code}\n```")
        lines.append(f"**Performance:** Executed in {sim.execution_time_ms}ms, produced {sim.output_rows} rows")
        lines.append(f"**Reuse Count:** Used successfully {sim.reuse_count} times\n")
    
    lines.append(
        "💡 **How to use this memory:**\n"
        "- If the current request is very similar, you can adapt the code patterns\n"
        "- Focus on column selection logic and transformation techniques\n"
        "- Ensure column names match the NEW schema (not the memory schema)\n"
        "- Improve upon the pattern if you see opportunities\n"
    )
    
    return "\n".join(lines)


def build_user_prompt(
    user_prompt: str,
    schema_description: str,
    row_count: int,
    similar_codes: list | None = None,  # RAG context
    include_examples: bool = True,
    multi_table_schemas: dict[str, str] | None = None,  # {table_name: schema_desc}
) -> str:
    """Build complete user prompt with optional RAG context and multi-table support."""
    parts = []

    # Add RAG context first if available (most important)
    if similar_codes:
        parts.append(build_rag_context_section(similar_codes))
        parts.append("")  # Blank line

    if include_examples:
        parts.append(build_few_shot_section())

    # Choose template based on single vs multi-table
    if multi_table_schemas and len(multi_table_schemas) > 1:
        # Multi-table mode
        tables_section = build_multi_table_section(multi_table_schemas)
        df_names = ", ".join(f"df_{name}" for name in multi_table_schemas.keys())

        parts.append(MULTI_TABLE_PROMPT_TEMPLATE.format(
            tables_section=tables_section,
            user_prompt=user_prompt,
            df_names=df_names,
        ))
    else:
        # Single table mode (original behavior)
        parts.append(USER_PROMPT_TEMPLATE.format(
            schema=schema_description,
            row_count=row_count,
            user_prompt=user_prompt,
        ))

    return "\n".join(parts)


def build_multi_table_section(table_schemas: dict[str, str]) -> str:
    """Build schema section for multiple tables."""
    lines = []
    for table_name, schema_desc in table_schemas.items():
        df_var = f"df_{table_name}"
        lines.append(f"### Table: `{df_var}`")
        lines.append(f"```")
        lines.append(schema_desc)
        lines.append(f"```")
        lines.append("")
    return "\n".join(lines)


def build_error_recovery_prompt(
    user_prompt: str,
    schema_description: str,
    previous_code: str,
    error_message: str,
) -> str:
    """Build error recovery prompt (legacy version)."""
    return ERROR_RECOVERY_TEMPLATE.format(
        error_message=error_message,
        previous_code=previous_code,
        schema=schema_description,
        user_prompt=user_prompt,
    )


def build_error_recovery_prompt_v2(
    user_prompt: str,
    schema_description: str,
    previous_code: str,
    error_message: str,
    error_analysis: "ErrorAnalysis | None" = None,
    multi_table_schemas: dict[str, str] | None = None,
) -> str:
    """
    Build intelligent error recovery prompt (V2).

    Uses structured error analysis to provide targeted fix instructions.

    Args:
        user_prompt: Original user request
        schema_description: Schema for single-table mode
        previous_code: The code that failed
        error_message: Raw error message
        error_analysis: Structured error analysis from ErrorAnalyzer
        multi_table_schemas: Dict of {table_name: schema_desc} for multi-table mode

    Returns:
        Formatted prompt for LLM recovery
    """
    # Build schema section (multi-table or single)
    if multi_table_schemas and len(multi_table_schemas) > 1:
        schema_section = "## Available Tables\n\n" + build_multi_table_section(multi_table_schemas)
    else:
        schema_section = f"## Dataset Schema\n{schema_description}"

    # Get error analysis context
    if error_analysis:
        error_analysis_text = error_analysis.to_recovery_context()
        fix_key = error_analysis.error_type.value
    else:
        error_analysis_text = f"Error: {error_message[:300]}"
        fix_key = "unknown"

    # Get fix instructions for this error type
    fix_instructions = ERROR_FIX_INSTRUCTIONS.get(
        fix_key,
        ERROR_FIX_INSTRUCTIONS["unknown"]
    )

    # Truncate error message for readability (keep first part)
    error_short = error_message[:500]
    if len(error_message) > 500:
        error_short += "\n... (truncated)"

    return ERROR_RECOVERY_TEMPLATE_V2.format(
        error_analysis=error_analysis_text,
        error_message_short=error_short,
        previous_code=previous_code,
        schema_section=schema_section,
        user_prompt=user_prompt,
        fix_instructions=fix_instructions,
    )
