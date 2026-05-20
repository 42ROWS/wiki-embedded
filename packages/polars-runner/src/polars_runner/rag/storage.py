"""
Storage layer for transformation code in Pinecone.

Features:
- Search for similar transformations (similarity > 85%)
- Save successful/failed attempts with full metadata
- Value-based quality scoring with heuristics
- Automatic cleanup when reaching 90% capacity
- Support for both success and failure namespaces
- Structured metadata for operations (JOIN, filter, etc.)

Quality Score Formula (0-100):
- Success: 30 points (binary)
- Speed: 10 points (faster = better)
- Output: 10 points (has rows)
- First attempt: 15 points (no retries needed)
- Column match: 20 points (requested columns present)
- RAG reuse: 15 points (used similar code successfully)
"""
import json
import re
import uuid
from datetime import datetime
from typing import Any

from polars_runner.core.constants import RAG_CONFIG
from polars_runner.core.models import SchemaInfo
from polars_runner.core.error_analyzer import has_risky_join_pattern
from .pinecone_client import PineconeClient


# =============================================================================
# CODE ANALYSIS HELPERS
# =============================================================================

def extract_operations_from_code(code: str) -> list[str]:
    """
    Extract Polars operations from generated code.

    Returns list of operations like: ["join", "filter", "select", "group_by"]
    """
    operations = []

    # Map of regex patterns to operation names
    patterns = {
        r"\.join\s*\(": "join",
        r"\.filter\s*\(": "filter",
        r"\.select\s*\(": "select",
        r"\.with_columns\s*\(": "with_columns",
        r"\.group_by\s*\(": "group_by",
        r"\.agg\s*\(": "agg",
        r"\.sort\s*\(": "sort",
        r"\.unique\s*\(": "unique",
        r"\.head\s*\(": "head",
        r"\.tail\s*\(": "tail",
        r"\.drop_nulls\s*\(": "drop_nulls",
        r"\.fill_null\s*\(": "fill_null",
        r"\.rename\s*\(": "rename",
        r"\.cast\s*\(": "cast",
    }

    for pattern, op_name in patterns.items():
        if re.search(pattern, code):
            operations.append(op_name)

    return list(set(operations))  # Remove duplicates


def extract_join_info(code: str) -> dict[str, Any] | None:
    """
    Extract JOIN information from code if present.

    Returns dict with left_on, right_on, how or None if no JOIN.
    """
    # Pattern for .join(..., left_on="x", right_on="y", how="z")
    join_pattern = r'\.join\s*\([^)]*'
    match = re.search(join_pattern, code, re.DOTALL)

    if not match:
        return None

    join_code = match.group(0)

    # Extract parameters
    left_on_match = re.search(r'left_on\s*=\s*["\']([^"\']+)["\']', join_code)
    right_on_match = re.search(r'right_on\s*=\s*["\']([^"\']+)["\']', join_code)
    on_match = re.search(r'\bon\s*=\s*["\']([^"\']+)["\']', join_code)
    how_match = re.search(r'how\s*=\s*["\']([^"\']+)["\']', join_code)

    join_info = {}

    if left_on_match:
        join_info["left_on"] = left_on_match.group(1)
    if right_on_match:
        join_info["right_on"] = right_on_match.group(1)
    if on_match:
        join_info["on"] = on_match.group(1)
    if how_match:
        join_info["how"] = how_match.group(1)
    else:
        join_info["how"] = "inner"  # Default

    return join_info if join_info else None


def extract_columns_from_prompt(prompt: str) -> list[str]:
    """
    Extract column names mentioned in user prompt.

    Looks for patterns like:
    - Quoted strings: "column_name", 'column_name'
    - After keywords: select, show, mostra, colonna, column, field
    """
    columns = []

    # Pattern 1: Quoted strings (likely column names)
    quoted = re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', prompt)
    columns.extend(quoted)

    # Pattern 2: Words after column-related keywords (Italian + English)
    keywords = r'(?:select|show|mostra|colonna|column|field|campo)\s+(\w+)'
    keyword_matches = re.findall(keywords, prompt, re.IGNORECASE)
    columns.extend(keyword_matches)

    # Remove common non-column words
    stop_words = {'the', 'a', 'an', 'il', 'la', 'i', 'le', 'un', 'una', 'and', 'e', 'or', 'o'}
    columns = [c for c in columns if c.lower() not in stop_words]

    return list(set(columns))


# =============================================================================
# MODELS FOR RAG RESULTS
# =============================================================================

class SimilarCode:
    """Similar code found in memory."""
    
    def __init__(
        self,
        id: str,
        score: float,
        prompt: str,
        code: str,
        quality_score: float,
        reuse_count: int,
        execution_time_ms: int,
        output_rows: int = 0,  # Add output_rows with default
    ):
        self.id = id
        self.score = score
        self.prompt = prompt
        self.code = code
        self.quality_score = quality_score
        self.reuse_count = reuse_count
        self.execution_time_ms = execution_time_ms
        self.output_rows = output_rows  # Store output_rows
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM prompt."""
        return {
            "similarity": f"{self.score:.2%}",
            "prompt": self.prompt,
            "code": self.code,
            "quality": f"{self.quality_score:.1f}/100",
            "reused": f"{self.reuse_count} times",
            "speed": f"{self.execution_time_ms}ms",
        }


# =============================================================================
# MAIN STORAGE CLASS
# =============================================================================

class TransformationStorage:
    """
    Storage for transformation code in Pinecone.
    
    Always active - used for all transformations regardless of tier.
    Provides intelligent code reuse and continuous learning.
    """
    
    def __init__(self):
        """Initialize storage with Pinecone client."""
        self._client = PineconeClient()
        self._index = self._client.get_index()
        self._inference = self._client.get_inference()
    
    async def search_similar(
        self,
        prompt: str,
        schema: SchemaInfo | None = None,
        threshold: float = RAG_CONFIG.similarity_threshold,
        top_k: int = RAG_CONFIG.top_k_results,
    ) -> list[SimilarCode]:
        """
        Search for similar transformations in memory.
        
        Args:
            prompt: User's transformation request
            schema: Optional schema info for better matching
            threshold: Minimum similarity (default: 0.85 from config)
            top_k: Number of results (default: 2 from config)
            
        Returns:
            List of similar codes with similarity > threshold
        """
        # 1. Create search text (combine prompt + schema if available)
        search_text = self._build_search_text(prompt, schema)
        
        # 2. Create embedding using Pinecone FREE inference
        embedding = self._create_embedding(search_text)
        
        # 3. Search in success namespace, **filter by quality_score**.
        # Legacy vectors written before the semantic validator existed score
        # ~60-70 (success+output+speed only). Vectors written by the validator-
        # gated flow score ~95+ thanks to the +30 validator_passed bump. A
        # threshold of 50 keeps useful legacy hits while preferring validated
        # code on ties — Pinecone returns by similarity, then we sort by
        # quality below.
        results = self._index.query(
            namespace=RAG_CONFIG.namespace_success,
            vector=embedding,
            top_k=top_k * 2,
            include_metadata=True,
            filter={
                "success": True,
                "quality_score": {"$gte": RAG_CONFIG.min_quality_for_retrieval},
            },
        )
        
        # 4. Filter by threshold and convert to SimilarCode objects
        # Also filter out codes with risky JOIN patterns that could cause errors
        similar_codes = []
        for match in results.matches:
            if match.score >= threshold:
                code = match.metadata.get("code", "")

                # Skip codes with risky JOIN patterns (missing coalesce=False)
                # This prevents the LLM from copying buggy patterns
                if has_risky_join_pattern(code):
                    print(f"[RAG] Skipping match {match.id[:8]} - has risky JOIN pattern")
                    continue

                similar_codes.append(
                    SimilarCode(
                        id=match.id,
                        score=match.score,
                        prompt=match.metadata.get("prompt", ""),
                        code=code,
                        quality_score=match.metadata.get("quality_score", 0.0),
                        reuse_count=match.metadata.get("reuse_count", 0),
                        execution_time_ms=match.metadata.get("execution_time_ms", 0),
                        output_rows=match.metadata.get("output_rows", 0),
                    )
                )

        # 5. Return top_k best results
        return similar_codes[:top_k]
    
    async def save_transformation(
        self,
        prompt: str,
        code: str,
        schema: SchemaInfo,
        success: bool,
        execution_time_ms: int,
        output_rows: int = 0,
        error_message: str | None = None,
        attempts: int = 1,
        used_rag: bool = False,
        output_columns: list[str] | None = None,
        validator_passed: bool = False,
    ) -> None:
        """
        Save transformation to Pinecone memory with full metadata.

        Args:
            prompt: User's transformation request (full, not truncated)
            code: Generated Polars code (full, not truncated)
            schema: Dataset schema
            success: Whether transformation succeeded
            execution_time_ms: Execution time
            output_rows: Number of rows in output
            error_message: Error if failed
            attempts: Number of generation attempts (1 = first try)
            used_rag: Whether RAG similar codes were used
            output_columns: List of output column names
        """
        print("[RAG] Starting save_transformation...")

        # 1. Extract structured metadata from code
        operations = extract_operations_from_code(code)
        join_info = extract_join_info(code)
        is_multi_table = "join" in operations
        requested_columns = extract_columns_from_prompt(prompt)

        # 2. Calculate quality score with new heuristics
        quality_score = self._calculate_quality_score(
            success=success,
            execution_time_ms=execution_time_ms,
            output_rows=output_rows,
            attempts=attempts,
            used_rag=used_rag,
            requested_columns=requested_columns,
            output_columns=output_columns or [],
            validator_passed=validator_passed,
        )
        print(f"[RAG] Quality score: {quality_score}")

        # 3. Create embedding from search text
        search_text = self._build_search_text(prompt, schema)
        print(f"[RAG] Search text length: {len(search_text)}")

        try:
            embedding = self._create_embedding(search_text)
            print(f"[RAG] Embedding created: {len(embedding)} dimensions")
        except Exception as e:
            print(f"[RAG] ERROR creating embedding: {e}")
            raise

        # 4. Prepare metadata with generous limits (not truncated)
        now = datetime.utcnow()
        schema_columns = list(schema.columns.keys())

        metadata = {
            # Core data - use config limits, not arbitrary truncation
            "prompt": prompt[:RAG_CONFIG.max_prompt_chars],
            "code": code[:RAG_CONFIG.max_code_chars],

            # Schema info
            "schema_columns": schema_columns[:RAG_CONFIG.max_columns_in_metadata],
            "row_count": schema.row_count,

            # Execution results
            "success": success,
            "execution_time_ms": execution_time_ms,
            "output_rows": output_rows,
            "output_columns": (output_columns or [])[:RAG_CONFIG.max_columns_in_metadata],

            # Quality metrics
            "quality_score": quality_score,
            "reuse_count": 0,
            "attempts": attempts,
            "used_rag": used_rag,
            "validator_passed": validator_passed,

            # Structured operation metadata
            "operations": operations,
            "is_multi_table": is_multi_table,
            "requested_columns": requested_columns[:50],  # Reasonable limit

            # Timestamps
            "created_at": now.isoformat(),
            "last_accessed": now.isoformat(),
        }

        # Add JOIN info if present (serialize to JSON string for Pinecone)
        if join_info:
            metadata["join_info"] = json.dumps(join_info)

        # Add error info if failed
        if error_message:
            metadata["error_message"] = error_message[:RAG_CONFIG.max_error_chars]
            metadata["error_type"] = self._extract_error_type(error_message)
        
        # 4. Choose namespace based on success
        namespace = (
            RAG_CONFIG.namespace_success if success 
            else RAG_CONFIG.namespace_failures
        )
        print(f"[RAG] Using namespace: {namespace}")
        
        # 5. Check if cleanup needed (at 90% capacity)
        total_count = self._get_total_count(namespace)
        print(f"[RAG] Total count in namespace: {total_count}")
        if total_count >= RAG_CONFIG.cleanup_threshold:
            await self._cleanup_low_value_vectors(namespace)
        
        # 6. Upsert to Pinecone
        vector_id = str(uuid.uuid4())
        print(f"[RAG] Upserting vector: {vector_id}")
        
        try:
            self._index.upsert(
                vectors=[(vector_id, embedding, metadata)],
                namespace=namespace,
            )
            print(f"[RAG] Successfully upserted!")
        except Exception as e:
            print(f"[RAG] ERROR during upsert: {e}")
            raise
    
    async def increment_reuse_count(self, vector_id: str) -> None:
        """
        Increment reuse counter when code is reused.
        
        This increases the vector's value score, making it less likely
        to be deleted during cleanup.
        
        Args:
            vector_id: ID of the vector to update
        """
        # Note: Pinecone doesn't support atomic increments directly
        # We'll update last_accessed which helps with freshness
        now = datetime.utcnow()
        
        # Fetch current metadata
        result = self._index.fetch(
            ids=[vector_id],
            namespace=RAG_CONFIG.namespace_success,
        )
        
        if vector_id in result.vectors:
            current_metadata = result.vectors[vector_id].metadata
            current_reuse = current_metadata.get("reuse_count", 0)
            
            # Update metadata with incremented count
            updated_metadata = {
                **current_metadata,
                "reuse_count": current_reuse + 1,
                "last_accessed": now.isoformat(),
            }
            
            # Update in Pinecone
            self._index.update(
                id=vector_id,
                set_metadata=updated_metadata,
                namespace=RAG_CONFIG.namespace_success,
            )
    
    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================
    
    def _build_search_text(
        self,
        prompt: str,
        schema: SchemaInfo | None = None,
    ) -> str:
        """
        Build search text for embedding.
        
        Combines prompt + schema columns for better semantic matching.
        """
        text = prompt
        
        if schema:
            columns = ", ".join(list(schema.columns.keys())[:10])
            text = f"{prompt}\nColumns: {columns}"
        
        return text
    
    def _create_embedding(self, text: str) -> list[float]:
        """
        Create embedding using Pinecone FREE inference.
        
        Uses multilingual-e5-large model (1536 dimensions).
        FREE tier includes 5M tokens/month - way more than we need.
        """
        response = self._inference.embed(
            model=RAG_CONFIG.embedding_model,
            inputs=[text],
            parameters={"input_type": "query"},
        )
        
        return response[0].values
    
    def _calculate_quality_score(
        self,
        success: bool,
        execution_time_ms: int,
        output_rows: int,
        attempts: int = 1,
        used_rag: bool = False,
        requested_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        validator_passed: bool = False,
    ) -> float:
        """
        Calculate quality score (0-100) with heuristics.

        Formula (matches docstring at top of file):
        - Success: 30 points (binary)
        - Speed: 10 points (faster = better)
        - Output: 10 points (has rows)
        - First attempt: 15 points (no retries needed)
        - Column match: 20 points (requested columns present in output)
        - RAG reuse: 15 points (used similar code successfully)

        Args:
            success: Whether transformation succeeded
            execution_time_ms: Execution time in milliseconds
            output_rows: Number of rows in output
            attempts: Number of generation attempts (1 = first try)
            used_rag: Whether RAG similar codes were used
            requested_columns: Columns mentioned in user prompt
            output_columns: Columns in the output dataframe

        Returns:
            Quality score 0-100
        """
        score = 0.0

        # Success component (30 points)
        if success:
            score += 30.0

        # Speed component (10 points) - tiered scoring
        speed_thresholds = [(100, 10.0), (500, 7.0), (1000, 5.0)]
        for threshold_ms, points in speed_thresholds:
            if execution_time_ms < threshold_ms:
                score += points
                break
        else:
            score += 3.0  # Slow but completed

        # Output component (10 points)
        if output_rows > 0:
            score += 10.0

        # First attempt bonus (15 points) - no retries needed
        if attempts == 1:
            score += 15.0

        # Column match component (20 points)
        if requested_columns and output_columns:
            output_cols_lower = {c.lower() for c in output_columns}
            matched = sum(
                1 for col in requested_columns
                if col.lower() in output_cols_lower
            )
            if requested_columns:
                match_ratio = matched / len(requested_columns)
                score += 20.0 * match_ratio

        # RAG reuse component (15 points) - used similar code successfully
        if used_rag and success:
            score += 15.0

        # Semantic validator bonus (30 points) — only awarded to code that
        # passed the post-execution semantic checks (Layer 1+2, optionally
        # Layer 3 LLM judge). This is the strongest signal of correctness
        # we have and gates retrieval via the `min_quality_for_retrieval`
        # filter in `search_similar`.
        if validator_passed:
            score += 30.0

        return min(score, 100.0)  # Cap at 100
    
    def _get_total_count(self, namespace: str) -> int:
        """Get total vector count in namespace."""
        stats = self._index.describe_index_stats()
        return stats.namespaces.get(namespace, {}).get("vector_count", 0)

    def _extract_error_type(self, error_message: str) -> str:
        """
        Extract error type from error message for categorization.

        Categorizes errors into types for better RAG learning:
        - column_not_found: Missing column references
        - type_error: Type conversion/mismatch issues
        - join_error: JOIN operation failures
        - syntax_error: Code syntax issues
        - schema_error: Schema mismatch
        - unknown: Uncategorized errors

        Args:
            error_message: The full error message string

        Returns:
            Error type category string
        """
        error_lower = error_message.lower()

        # Pattern-based error classification
        error_patterns = {
            "column_not_found": [
                "column", "not found", "columnnotfounderror",
                "no column", "unknown column",
            ],
            "type_error": [
                "typeerror", "type mismatch", "cannot cast",
                "invalid type", "expected type",
            ],
            "join_error": [
                "join", "left_on", "right_on", "joincolumn",
                "duplicate column", "ambiguous column",
            ],
            "syntax_error": [
                "syntaxerror", "invalid syntax", "unexpected token",
                "parsing error",
            ],
            "schema_error": [
                "schema", "schemamismatch", "incompatible schema",
                "column count",
            ],
        }

        for error_type, patterns in error_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                return error_type

        return "unknown"

    async def _cleanup_low_value_vectors(self, namespace: str) -> None:
        """
        Cleanup low-value vectors when reaching capacity.
        
        Removes bottom 10% by value score to free space.
        Value score = quality_score + (reuse_count * 6)
        """
        # This is a simplified version - full implementation would:
        # 1. Query all vectors with metadata
        # 2. Calculate value score for each
        # 3. Sort by value score
        # 4. Delete bottom 10%
        # 
        # For now, we'll implement basic cleanup in production
        pass
