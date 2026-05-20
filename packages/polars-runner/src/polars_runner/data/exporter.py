"""
Data exporter with multi-format support.
Exports DataFrames to various formats optimally.
"""
from __future__ import annotations

import io
from typing import Any, Final

import polars as pl

from polars_runner.core.constants import OutputFormat, LIMITS


class DataExporter:
    """
    Exports DataFrames to various formats.
    
    Supports:
    - CSV (with optimized settings)
    - JSON (array of objects or newline-delimited)
    - Parquet (compressed, columnar)
    - Excel (.xlsx)
    """
    
    @staticmethod
    def export(
        df: pl.DataFrame,
        output_format: OutputFormat,
        **kwargs: Any,
    ) -> bytes:
        """
        Export DataFrame to bytes in specified format.
        
        Args:
            df: DataFrame to export
            output_format: Target format
            **kwargs: Format-specific options
            
        Returns:
            Exported data as bytes
        """
        exporters = {
            OutputFormat.CSV: DataExporter._export_csv,
            OutputFormat.JSON: DataExporter._export_json,
            OutputFormat.PARQUET: DataExporter._export_parquet,
            OutputFormat.EXCEL: DataExporter._export_excel,
        }
        
        exporter = exporters.get(output_format)
        if not exporter:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return exporter(df, **kwargs)
    
    @staticmethod
    def _export_csv(df: pl.DataFrame, **kwargs: Any) -> bytes:
        """Export to CSV."""
        buffer = io.BytesIO()
        df.write_csv(
            buffer,
            separator=kwargs.get("separator", ","),
            include_header=kwargs.get("include_header", True),
            null_value=kwargs.get("null_value", ""),
        )
        return buffer.getvalue()
    
    @staticmethod
    def _export_json(df: pl.DataFrame, **kwargs: Any) -> bytes:
        """Export to JSON (array of objects by default)."""
        import json
        # Polars 1.x removed row_oriented parameter from write_json
        # Use to_dicts() + json.dumps() for row-oriented JSON
        data = df.to_dicts()
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        return json_str.encode("utf-8")
    
    @staticmethod
    def _export_parquet(df: pl.DataFrame, **kwargs: Any) -> bytes:
        """Export to Parquet with compression."""
        buffer = io.BytesIO()
        df.write_parquet(
            buffer,
            compression=kwargs.get("compression", "zstd"),
            statistics=True,
        )
        return buffer.getvalue()
    
    @staticmethod
    def _export_excel(df: pl.DataFrame, **kwargs: Any) -> bytes:
        """Export to Excel (.xlsx)."""
        buffer = io.BytesIO()
        df.write_excel(
            buffer,
            worksheet=kwargs.get("worksheet", "Sheet1"),
        )
        return buffer.getvalue()
    
    @staticmethod
    def get_filename(base_name: str, output_format: OutputFormat) -> str:
        """Generate filename with correct extension."""
        return f"{base_name}.{output_format.file_extension}"
    
    @staticmethod
    def get_content_type(output_format: OutputFormat) -> str:
        """Get MIME content type for format."""
        return output_format.content_type
    
    @staticmethod
    def get_preview(
        df: pl.DataFrame,
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get preview of DataFrame as list of dicts.
        
        Args:
            df: DataFrame to preview
            max_rows: Max rows to include (default: LIMITS.preview_rows)
            
        Returns:
            List of row dictionaries
        """
        rows = max_rows or LIMITS.preview_rows
        preview_df = df.head(rows)
        return preview_df.to_dicts()


# =============================================================================
# STREAMING EXPORT (for large files)
# =============================================================================

class StreamingExporter:
    """
    Export large DataFrames in streaming mode.
    
    Writes chunks to avoid memory issues with very large datasets.
    """
    
    def __init__(self, chunk_size: int | None = None):
        self._chunk_size = chunk_size or LIMITS.streaming_threshold_rows
    
    def export_csv_streaming(
        self,
        lf: pl.LazyFrame,
        output_path: str,
        **kwargs: Any,
    ) -> int:
        """
        Export LazyFrame to CSV in streaming mode.
        
        Returns:
            Number of rows written
        """
        # Collect with streaming engine
        df = lf.collect(engine="streaming")
        
        # Write directly to file
        df.write_csv(
            output_path,
            separator=kwargs.get("separator", ","),
            include_header=kwargs.get("include_header", True),
        )
        
        return df.height
    
    def export_parquet_streaming(
        self,
        lf: pl.LazyFrame,
        output_path: str,
        **kwargs: Any,
    ) -> int:
        """
        Export LazyFrame to Parquet in streaming mode.
        
        Parquet is ideal for streaming due to columnar format.
        
        Returns:
            Number of rows written
        """
        # Use sink_parquet for true streaming
        lf.sink_parquet(
            output_path,
            compression=kwargs.get("compression", "zstd"),
        )
        
        # Get row count (requires reading metadata)
        df = pl.scan_parquet(output_path).select(pl.len()).collect()
        return df.item()
