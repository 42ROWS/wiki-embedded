"""
Data loader with optimized multi-file support.
Uses Polars lazy evaluation for maximum performance.
"""
import io
import time
from typing import Any, Iterator
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import httpx

from polars_runner.core.constants import (
    InputFormat,
    DataSourceType,
    LIMITS,
    POLARS_CONFIG,
)
from polars_runner.core.exceptions import (
    DataLoadingError,
    SchemaMismatchError,
    ValidationError,
)
from polars_runner.core.models import (
    DataSource,
    SchemaInfo,
    LoadedDataset,
    MergedDatasetInfo,
)


class DataLoader:
    """
    High-performance data loader with lazy evaluation.
    
    Strategy for file multipli:
    1. Load each file as LazyFrame (no memory until needed)
    2. Validate schemas for compatibility
    3. Concat with automatic type coercion
    4. Collect only when transformation runs
    """
    
    def __init__(self, http_timeout: int = 60):
        self._http_client = httpx.Client(timeout=http_timeout, follow_redirects=True)
        self._loaded_frames: dict[str, pl.LazyFrame] = {}
    
    def __del__(self):
        self._http_client.close()
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def load_sources(
        self,
        sources: tuple[DataSource, ...],
        validate_schemas: bool = True,
    ) -> tuple[pl.LazyFrame, MergedDatasetInfo]:
        """
        Load multiple sources and merge into single LazyFrame.
        
        Returns:
            Tuple of (merged LazyFrame, merge info)
        """
        if not sources:
            raise ValidationError("No data sources provided", field="data_sources")
        
        if len(sources) > LIMITS.max_files_count:
            raise ValidationError(
                f"Too many files: {len(sources)} (max: {LIMITS.max_files_count})",
                field="data_sources",
            )
        
        # Load each source
        loaded: list[LoadedDataset] = []
        lazy_frames: list[pl.LazyFrame] = []
        warnings: list[str] = []
        
        for source in sources:
            start = time.perf_counter()
            lf, schema = self._load_single_source(source)
            load_time_ms = int((time.perf_counter() - start) * 1000)
            
            # Get metadata without collecting
            row_count = self._estimate_row_count(lf)
            size_mb = self._estimate_size_mb(lf, schema)
            
            loaded.append(LoadedDataset(
                source=source,
                schema=schema,
                row_count=row_count,
                estimated_size_mb=size_mb,
                load_time_ms=load_time_ms,
            ))
            lazy_frames.append(lf)
        
        # Validate schema compatibility if multiple files
        if validate_schemas and len(loaded) > 1:
            warnings.extend(self._validate_schema_compatibility(loaded))
        
        # Merge strategy
        if len(lazy_frames) == 1:
            merged_lf = lazy_frames[0]
            merge_strategy = "single_source"
        else:
            merged_lf, merge_strategy = self._merge_lazy_frames(lazy_frames, loaded)
        
        # Build unified schema
        unified_schema = self._build_unified_schema(merged_lf, loaded)
        
        merge_info = MergedDatasetInfo(
            sources=tuple(loaded),
            total_rows=sum(l.row_count for l in loaded),
            unified_schema=unified_schema,
            merge_strategy=merge_strategy,
            warnings=warnings,
        )
        
        return merged_lf, merge_info
    
    def load_single(self, source: DataSource) -> tuple[pl.LazyFrame, SchemaInfo]:
        """Load a single source as LazyFrame."""
        return self._load_single_source(source)

    def load_sources_separate(
        self,
        sources: tuple[DataSource, ...],
    ) -> dict[str, tuple[pl.LazyFrame, SchemaInfo]]:
        """
        Load multiple sources as separate tables (for multi-table JOIN operations).

        Unlike load_sources() which merges all sources, this keeps them separate.
        Used when sources have different schemas and user wants to JOIN them.

        Returns:
            Dict mapping table_name -> (LazyFrame, SchemaInfo)
            Table names are derived from DataSource.table_name or generated.
        """
        if not sources:
            raise ValidationError("No data sources provided", field="data_sources")

        result: dict[str, tuple[pl.LazyFrame, SchemaInfo]] = {}

        for i, source in enumerate(sources):
            lf, schema = self._load_single_source(source)

            # Determine table name
            if source.table_name:
                table_name = source.table_name
            else:
                # Generate name from location or index
                if source.source_type == DataSourceType.INLINE:
                    table_name = f"table_{i + 1}"
                else:
                    # Extract filename from URL/path
                    name = source.location.split("/")[-1].split("?")[0]
                    name = name.rsplit(".", 1)[0]  # Remove extension
                    table_name = name if name else f"table_{i + 1}"

            # Ensure unique name
            base_name = table_name
            counter = 2
            while table_name in result:
                table_name = f"{base_name}_{counter}"
                counter += 1

            result[table_name] = (lf, schema)

        return result
    
    # =========================================================================
    # PRIVATE - Loading
    # =========================================================================
    
    def _load_single_source(
        self,
        source: DataSource,
    ) -> tuple[pl.LazyFrame, SchemaInfo]:
        """Load single source based on type."""
        
        match source.source_type:
            case DataSourceType.URL | DataSourceType.UPLOAD:
                return self._load_from_url(source)
            case DataSourceType.APIFY_DATASET:
                return self._load_from_apify_dataset(source)
            case DataSourceType.INLINE:
                return self._load_from_inline(source)
            case _:
                raise ValidationError(
                    f"Unknown source type: {source.source_type}",
                    field="source_type",
                )
    
    def _load_from_url(
        self,
        source: DataSource,
    ) -> tuple[pl.LazyFrame, SchemaInfo]:
        """Load data from URL (HTTP/HTTPS)."""
        url = source.location
        
        # Fix Google Drive URLs for large files (>25MB trigger virus scan)
        url = self._fix_google_drive_url(url)
        
        try:
            # Fetch content
            response = self._http_client.get(url)
            response.raise_for_status()
            content = response.content
            
        except httpx.HTTPError as e:
            raise DataLoadingError(
                f"Failed to fetch URL: {url}",
                source=url,
                original_error=e,
            )
        
        # Check if response is HTML (Google Drive virus scan warning)
        if self._is_html_response(content):
            raise DataLoadingError(
                "Received HTML instead of data file. "
                "This usually happens with Google Drive files >25MB. "
                "Try using a direct download link or different hosting.",
                source=url,
            )
        
        # Extract from ZIP if necessary
        content, extracted_filename = self._extract_from_zip_if_needed(content)
        
        # Detect format (prefer extracted filename if available)
        if extracted_filename:
            file_format = self._detect_format_from_url(extracted_filename)
        else:
            file_format = source.format or self._detect_format_from_url(url)
        
        # Parse based on format
        return self._parse_content(content, file_format, source.location)
    
    def _fix_google_drive_url(self, url: str) -> str:
        """Fix Google Drive URLs to bypass virus scan warning for large files."""
        import re
        
        # Check if it's a Google Drive URL
        if "drive.google.com" not in url:
            return url
        
        # Extract file ID from various Google Drive URL formats
        file_id = None
        
        # Format: /uc?id=FILE_ID
        match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
        if match:
            file_id = match.group(1)
        
        # Format: /file/d/FILE_ID/
        if not file_id:
            match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
            if match:
                file_id = match.group(1)
        
        # Format: /open?id=FILE_ID
        if not file_id:
            match = re.search(r'/open\?id=([a-zA-Z0-9_-]+)', url)
            if match:
                file_id = match.group(1)
        
        if not file_id:
            # Can't extract ID, return original with confirm=t
            separator = "&" if "?" in url else "?"
            return f"{url}{separator}confirm=t"
        
        # Use the direct download URL format that bypasses virus scan
        # This format works reliably for large files
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    
    def _is_html_response(self, content: bytes) -> bool:
        """Check if response content is HTML instead of expected data."""
        # Check first 500 bytes for HTML signatures
        header = content[:500].lower()
        html_signatures = [
            b"<!doctype html",
            b"<html",
            b"<head",
            b"virus scan warning",
            b"google drive",
        ]
        return any(sig in header for sig in html_signatures)
    
    def _extract_from_zip_if_needed(self, content: bytes) -> tuple[bytes, str | None]:
        """
        Extract data file from ZIP archive if content is a ZIP file.
        
        Returns:
            Tuple of (extracted_content, filename) or (original_content, None)
        """
        import zipfile
        
        # Check ZIP magic bytes (PK\x03\x04)
        if len(content) < 4 or content[:4] != b'PK\x03\x04':
            return content, None
        
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Get list of files in archive
                file_list = zf.namelist()
                
                # Priority: CSV > JSON > Parquet > Excel > first file
                data_extensions = ['.csv', '.json', '.parquet', '.xlsx', '.xls']
                
                for ext in data_extensions:
                    for filename in file_list:
                        if filename.lower().endswith(ext) and not filename.startswith('__MACOSX'):
                            extracted = zf.read(filename)
                            return extracted, filename
                
                # Fallback: extract first non-directory file
                for filename in file_list:
                    if not filename.endswith('/') and not filename.startswith('__MACOSX'):
                        extracted = zf.read(filename)
                        return extracted, filename
                
                raise DataLoadingError(
                    "ZIP archive is empty or contains no data files",
                    source="zip_extraction",
                )
                
        except zipfile.BadZipFile:
            # Not a valid ZIP, return original content
            return content, None
    
    def _load_from_apify_dataset(
        self,
        source: DataSource,
    ) -> tuple[pl.LazyFrame, SchemaInfo]:
        """Load from Apify dataset ID."""
        import os
        
        dataset_id = source.location
        token = os.getenv("APIFY_TOKEN")
        
        if not token:
            raise ValidationError(
                "APIFY_TOKEN required for loading Apify datasets",
                field="apifyDatasetId",
            )
        
        # Fetch dataset items via API
        url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=json"
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = self._http_client.get(url, headers=headers)
            response.raise_for_status()
            items = response.json()
            
        except httpx.HTTPError as e:
            raise DataLoadingError(
                f"Failed to load Apify dataset: {dataset_id}",
                source=dataset_id,
                original_error=e,
            )
        
        if not items:
            raise DataLoadingError(
                f"Apify dataset is empty: {dataset_id}",
                source=dataset_id,
            )
        
        # Convert to LazyFrame
        df = pl.DataFrame(items)
        schema = self._extract_schema(df)

        return df.lazy(), schema

    def _load_from_inline(
        self,
        source: DataSource,
    ) -> tuple[pl.LazyFrame, SchemaInfo]:
        """Load from inline JSON data (zero I/O)."""
        data = source.inline_data

        if data is None:
            raise ValidationError(
                "Inline data source has no data",
                field="inputData",
            )

        # Handle different input formats
        if isinstance(data, list):
            # Array of objects: [{"col": "val"}, ...]
            if not data:
                raise DataLoadingError(
                    "inputData array is empty",
                    source="inline",
                )
            items = data
        elif isinstance(data, dict):
            # Could be single object or named tables
            # For now, treat as single table if values are not lists
            first_value = next(iter(data.values()), None)
            if isinstance(first_value, list):
                # Named tables: {"table1": [...], "table2": [...]}
                # For now, use first table (multi-table support can be added later)
                table_name = next(iter(data.keys()))
                items = data[table_name]
                if not items:
                    raise DataLoadingError(
                        f"inputData table '{table_name}' is empty",
                        source="inline",
                    )
            else:
                # Single object: {"col": "val"} -> wrap in array
                items = [data]
        else:
            raise ValidationError(
                f"inputData must be array or object, got {type(data).__name__}",
                field="inputData",
            )

        # Convert to DataFrame
        df = pl.DataFrame(items)
        schema = self._extract_schema(df)

        return df.lazy(), schema

    def _parse_content(
        self,
        content: bytes,
        file_format: InputFormat,
        source_name: str,
    ) -> tuple[pl.LazyFrame, SchemaInfo]:
        """Parse content based on format."""
        
        try:
            match file_format:
                case InputFormat.CSV:
                    df = self._parse_csv(content)
                case InputFormat.JSON:
                    df = self._parse_json(content)
                case InputFormat.PARQUET:
                    df = self._parse_parquet(content)
                case InputFormat.EXCEL | InputFormat.EXCEL_OLD:
                    df = self._parse_excel(content)
                case _:
                    raise ValidationError(
                        f"Unsupported format: {file_format}",
                        field="format",
                    )
            
            schema = self._extract_schema(df)
            return df.lazy(), schema
            
        except pl.exceptions.PolarsError as e:
            raise DataLoadingError(
                f"Failed to parse {file_format.value} content",
                source=source_name,
                original_error=e,
            )
    
    # =========================================================================
    # PRIVATE - Format Parsers
    # =========================================================================
    
    def _parse_csv(self, content: bytes) -> pl.DataFrame:
        """Parse CSV with intelligent inference."""
        # Try to detect encoding
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1
            text = content.decode("latin-1")
        
        # Detect separator
        separator = self._detect_csv_separator(text[:2000])
        
        return pl.read_csv(
            io.StringIO(text),
            separator=separator,
            infer_schema_length=POLARS_CONFIG.infer_schema_length,
            null_values=list(POLARS_CONFIG.null_values),
            try_parse_dates=True,
            ignore_errors=False,
        )
    
    def _parse_json(self, content: bytes) -> pl.DataFrame:
        """Parse JSON (array of objects or newline-delimited)."""
        text = content.decode("utf-8")
        
        # Try array of objects first
        if text.strip().startswith("["):
            return pl.read_json(io.BytesIO(content))
        
        # Try newline-delimited JSON
        return pl.read_ndjson(io.BytesIO(content))
    
    def _parse_parquet(self, content: bytes) -> pl.DataFrame:
        """Parse Parquet."""
        return pl.read_parquet(io.BytesIO(content))
    
    def _parse_excel(self, content: bytes) -> pl.DataFrame:
        """Parse Excel (xlsx/xls)."""
        return pl.read_excel(io.BytesIO(content))
    
    # =========================================================================
    # PRIVATE - Schema Operations
    # =========================================================================
    
    def _extract_schema(self, df: pl.DataFrame) -> SchemaInfo:
        """Extract schema information from DataFrame."""
        columns = {col: str(dtype) for col, dtype in df.schema.items()}
        
        # Sample values (first N non-null)
        sample_values = {}
        for col in df.columns[:LIMITS.max_columns]:
            vals = (
                df.select(col)
                .drop_nulls()
                .head(5)
                .to_series()
                .to_list()
            )
            sample_values[col] = vals
        
        # Null counts
        null_counts = {
            col: df.select(pl.col(col).null_count()).item()
            for col in df.columns
        }
        
        return SchemaInfo(
            columns=columns,
            row_count=df.height,
            null_counts=null_counts,
            sample_values=sample_values,
        )
    
    def _validate_schema_compatibility(
        self,
        datasets: list[LoadedDataset],
    ) -> list[str]:
        """Validate schemas are compatible for merging."""
        warnings = []
        
        if len(datasets) < 2:
            return warnings
        
        base_schema = datasets[0].schema
        base_columns = set(base_schema.columns.keys())
        
        for i, dataset in enumerate(datasets[1:], 2):
            current_columns = set(dataset.schema.columns.keys())
            
            # Check for missing columns
            missing = base_columns - current_columns
            extra = current_columns - base_columns
            
            if missing:
                warnings.append(
                    f"File {i} missing columns: {missing}. Will be filled with null."
                )
            if extra:
                warnings.append(
                    f"File {i} has extra columns: {extra}. Will be included."
                )
            
            # Check type mismatches for common columns
            common = base_columns & current_columns
            for col in common:
                base_type = base_schema.columns[col]
                current_type = dataset.schema.columns[col]
                if base_type != current_type:
                    warnings.append(
                        f"Column '{col}' type mismatch: {base_type} vs {current_type}. "
                        "Will attempt automatic coercion."
                    )
        
        return warnings
    
    def _build_unified_schema(
        self,
        lf: pl.LazyFrame,
        datasets: list[LoadedDataset],
    ) -> SchemaInfo:
        """Build unified schema from merged LazyFrame."""
        # Collect small sample for schema
        sample_df = lf.head(LIMITS.schema_sample_rows).collect()
        return self._extract_schema(sample_df)
    
    # =========================================================================
    # PRIVATE - Merge Operations
    # =========================================================================
    
    def _merge_lazy_frames(
        self,
        frames: list[pl.LazyFrame],
        datasets: list[LoadedDataset],
    ) -> tuple[pl.LazyFrame, str]:
        """
        Merge multiple LazyFrames.
        
        Strategy:
        - Vertical concat (stack rows) if schemas are compatible
        - Uses diagonal concat to handle different column sets
        """
        # Get all unique columns
        all_columns: set[str] = set()
        for dataset in datasets:
            all_columns.update(dataset.schema.columns.keys())
        
        # Use diagonal concat - handles different schemas gracefully
        merged = pl.concat(frames, how="diagonal_relaxed")
        
        # Note: rechunk() is only available on DataFrame, not LazyFrame
        # The rechunk will happen automatically when collect() is called
        # or we can add .collect().lazy() if needed for performance
        
        return merged, "concat_diagonal"
    
    # =========================================================================
    # PRIVATE - Utilities
    # =========================================================================
    
    def _detect_format_from_url(self, url: str) -> InputFormat:
        """Detect file format from URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Remove query params
        ext = path.split(".")[-1].split("?")[0]
        
        try:
            return InputFormat.from_extension(ext)
        except ValueError:
            # Default to CSV
            return InputFormat.CSV
    
    def _detect_csv_separator(self, sample: str) -> str:
        """Detect CSV separator from sample."""
        candidates = [",", ";", "\t", "|"]
        counts = {sep: sample.count(sep) for sep in candidates}
        return max(counts, key=counts.get)
    
    def _estimate_row_count(self, lf: pl.LazyFrame) -> int:
        """Estimate row count without full collection."""
        # For lazy frames, we need to collect to count
        # Use optimized counting
        try:
            return lf.select(pl.len()).collect().item()
        except Exception:
            return -1  # Unknown
    
    def _estimate_size_mb(self, lf: pl.LazyFrame, schema: SchemaInfo) -> float:
        """Estimate data size in MB."""
        # Rough estimation based on schema and row count
        bytes_per_row = sum(
            self._dtype_size_bytes(dtype)
            for dtype in schema.columns.values()
        )
        return (schema.row_count * bytes_per_row) / (1024 * 1024)
    
    def _dtype_size_bytes(self, dtype_str: str) -> int:
        """Estimate bytes per value for dtype."""
        dtype_lower = dtype_str.lower()
        if "int8" in dtype_lower or "bool" in dtype_lower:
            return 1
        elif "int16" in dtype_lower:
            return 2
        elif "int32" in dtype_lower or "float32" in dtype_lower:
            return 4
        elif "int64" in dtype_lower or "float64" in dtype_lower:
            return 8
        elif "str" in dtype_lower or "utf8" in dtype_lower:
            return 50  # Average string length estimate
        else:
            return 8  # Default
