"""
AI Data Transformer - Main entry point.
Orchestrates the complete transformation pipeline.

Pricing Model:
- BYOK (Bring Your Own Key): User provides LLM API key → "transformation_byok" event
- Hosted: We provide Gemini Flash-Lite → "transformation_hosted" event
"""
import os
import time
import json
import asyncio
from typing import Any

import polars as pl

from polars_runner.core.constants import (
    LLMProvider,
    OutputFormat,
    ExecutionStatus,
    PricingEvent,
    LIMITS,
    FILE_PATTERNS,
    ORACLE_CONFIG,
)
from polars_runner.core.models import (
    TransformationInput,
    TransformationResult,
    SchemaInfo,
)
from polars_runner.core.exceptions import (
    ValidationError,
    TransformerError,
)
from polars_runner.core.executor import execute_transformation, execute_multi_table_transformation
from polars_runner.core.error_analyzer import ErrorAnalyzer
from polars_runner.data.loader import DataLoader
from polars_runner.data.exporter import DataExporter
from polars_runner.executor.oracle import (
    Oracle,
    generate_oracle,
    verify_against_oracle,
)
from polars_runner.executor.semantic_validator import (
    SemanticValidationFailure,
    SemanticVerdict,
    VerdictLevel,
    validate_semantic,
)
from polars_runner.llm.client import LLMService
from polars_runner.rag.storage import TransformationStorage


# ---------------------------------------------------------------------------
# Oracle LLM adapter — implements `executor.oracle.LLMClient` Protocol on
# top of the `google-genai` SDK. Used only when ORACLE_CONFIG.enabled.
# ---------------------------------------------------------------------------

class _GeminiOracleClient:
    """Minimal :class:`LLMClient` implementation for oracle generation.

    The oracle pipeline is provider-agnostic by design (see Protocol in
    `executor.oracle`); this adapter is the default backed by Gemini, which
    matches the rest of the actor's premium-tier choices. To add Claude /
    OpenAI, implement the same three-argument ``complete()`` signature.
    """

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def complete(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response_mime_type: str | None = None,
    ) -> str:
        from google.genai import types as gtypes
        # NOTE: Gemini 2.5 Pro requires thinking mode — `thinking_budget=0`
        # is rejected with HTTP 400. We leave `thinking_config` unset so the
        # SDK applies the model's default budget. This is the only correct
        # configuration for Pro-tier oracle generation.
        cfg = gtypes.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type=response_mime_type,
        )
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg,
        )
        return response.text or ""


# =============================================================================
# APIFY ADAPTER
# =============================================================================

class ApifyAdapter:
    """
    Adapter for Apify SDK.
    Falls back to local mock when not on Apify platform.
    """
    
    def __init__(self):
        # APIFY_IS_AT_HOME is set to "1" on Apify platform
        apify_env = os.getenv("APIFY_IS_AT_HOME", "")
        print(f"[DEBUG] APIFY_IS_AT_HOME = '{apify_env}'")
        self._is_apify = apify_env.lower() in ("true", "1", "yes")
        print(f"[DEBUG] _is_apify = {self._is_apify}")
        self._actor = None
    
    async def __aenter__(self):
        if self._is_apify:
            from apify import Actor
            self._actor = Actor
            await Actor.init()
        return self
    
    async def __aexit__(self, *args):
        if self._is_apify and self._actor:
            await self._actor.exit()
    
    async def get_input(self) -> dict[str, Any]:
        if self._is_apify:
            return await self._actor.get_input() or {}
        # Local: read from file or env
        return self._load_local_input()
    
    def _load_local_input(self) -> dict[str, Any]:
        import json
        input_file = os.getenv("INPUT_FILE", "docker/data/input.json")
        if os.path.exists(input_file):
            with open(input_file) as f:
                return json.load(f)
        return {}
    
    def log_info(self, msg: str) -> None:
        if self._is_apify:
            self._actor.log.info(msg)
        else:
            print(f"[INFO] {msg}")
    
    def log_warning(self, msg: str) -> None:
        if self._is_apify:
            self._actor.log.warning(msg)
        else:
            print(f"[WARN] {msg}")
    
    def log_error(self, msg: str) -> None:
        if self._is_apify:
            self._actor.log.error(msg)
        else:
            print(f"[ERROR] {msg}")
    
    async def set_value(self, key: str, value: Any, content_type: str | None = None) -> None:
        if self._is_apify:
            await self._actor.set_value(key, value, content_type=content_type)
        else:
            # Local: save to output directory
            output_dir = os.getenv("OUTPUT_DIR", "output")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, key)
            if isinstance(value, bytes):
                with open(path, "wb") as f:
                    f.write(value)
            else:
                with open(path, "w") as f:
                    f.write(str(value))
            print(f"[OUTPUT] Saved: {path}")
    
    async def push_data(self, data: dict[str, Any]) -> None:
        if self._is_apify:
            await self._actor.push_data(data)
        else:
            import json
            output_dir = os.getenv("OUTPUT_DIR", "output")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "result.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"[OUTPUT] Result saved: {path}")
    
    async def charge_event(self, event_name: str, count: int = 1, event_data: dict[str, Any] | None = None) -> None:
        """
        Charge for a pricing event using Apify pay-per-event model.
        
        Events must be configured in Apify Console under Actor pricing:
        - transformation_byok: User provided their own API key
        - transformation_hosted: Using our hosted Gemini Flash-Lite
        
        Args:
            event_name: Name of the event to charge for (must match Apify Console config)
            count: Number of events to charge for (default: 1)
            event_data: Optional metadata for logging (not sent to Apify)
        """
        if self._is_apify:
            try:
                # Apify SDK uses Actor.charge() for pay-per-event billing
                await self._actor.charge(event_name=event_name, count=count)
                self.log_info(f"💰 Charged event: {event_name} (count={count})")
            except AttributeError:
                # Actor.charge() not available - older SDK version
                self.log_warning(f"Event charging not available (SDK too old): {event_name}")
            except Exception as e:
                # Charging failed - log but don't fail the transformation
                # This can happen if PPE not configured in Apify Console
                self.log_warning(f"Event charging failed for '{event_name}': {e}")
        else:
            print(f"[EVENT] {event_name} (count={count}): {event_data}")


# =============================================================================
# MAIN TRANSFORMER
# =============================================================================

class Transformer:
    """
    Main transformation orchestrator.
    Coordinates loading, generation, execution, and export.
    Now includes RAG (Retrieval-Augmented Generation) for code reuse.
    """
    
    def __init__(self, adapter: ApifyAdapter):
        self._adapter = adapter
        self._loader = DataLoader()
        self._rag_storage = None  # Lazy init (optional dependency)
    
    def _init_rag_storage(self) -> TransformationStorage | None:
        """Lazy initialize RAG storage (optional)."""
        if self._rag_storage is not None:
            return self._rag_storage
        
        try:
            self._rag_storage = TransformationStorage()
            return self._rag_storage
        except Exception as e:
            self._adapter.log_warning(
                f"RAG storage initialization failed: {e}. "
                "Continuing without code reuse."
            )
            return None
    
    async def run(self, raw_input: dict[str, Any]) -> TransformationResult:
        """Execute complete transformation pipeline."""
        start_time = time.perf_counter()
        warnings: list[str] = []
        errors: list[dict[str, Any]] = []
        
        # 1. Parse and validate input
        self._adapter.log_info("Parsing input...")
        try:
            transformation_input = TransformationInput.from_actor_input(raw_input)
        except Exception as e:
            raise ValidationError(f"Invalid input: {e}")
        
        self._validate_input(transformation_input)
        
        # 2. Load data sources - detect single vs multi-table mode
        self._adapter.log_info(
            f"Loading {len(transformation_input.data_sources)} data source(s)..."
        )

        # Check if this is multi-table mode (sources have explicit table_name)
        is_multi_table = any(
            src.table_name is not None
            for src in transformation_input.data_sources
        )

        # Storage for multi-table mode
        multi_table_schemas: dict[str, SchemaInfo] | None = None
        extra_dataframes: dict[str, pl.LazyFrame] | None = None
        merge_info = None

        if is_multi_table:
            # Multi-table mode: load each table separately for JOIN operations
            self._adapter.log_info("📊 Multi-table mode detected - keeping tables separate for JOIN")

            tables = self._loader.load_sources_separate(transformation_input.data_sources)

            # Build multi-table schemas dict
            multi_table_schemas = {}
            extra_dataframes = {}
            total_rows = 0
            all_columns = []

            # Use first table as primary df
            table_names = list(tables.keys())
            primary_name = table_names[0]
            lazy_frame, primary_schema = tables[primary_name]

            multi_table_schemas[primary_name] = primary_schema
            total_rows += primary_schema.row_count
            all_columns.extend(primary_schema.columns.keys())

            self._adapter.log_info(
                f"  - df_{primary_name}: {primary_schema.row_count:,} rows, "
                f"{len(primary_schema.columns)} columns"
            )

            # Add remaining tables as extra dataframes
            for table_name in table_names[1:]:
                lf, schema = tables[table_name]
                extra_dataframes[table_name] = lf
                multi_table_schemas[table_name] = schema
                total_rows += schema.row_count
                all_columns.extend(schema.columns.keys())

                self._adapter.log_info(
                    f"  - df_{table_name}: {schema.row_count:,} rows, "
                    f"{len(schema.columns)} columns"
                )

            # Create a synthetic merge_info for compatibility
            from polars_runner.core.models import MergedDatasetInfo, LoadedDataset
            merge_info = MergedDatasetInfo(
                sources=tuple(
                    LoadedDataset(
                        source=transformation_input.data_sources[i],
                        schema=list(multi_table_schemas.values())[i],
                        row_count=list(multi_table_schemas.values())[i].row_count,
                        estimated_size_mb=0.0,
                        load_time_ms=0,
                    )
                    for i in range(len(table_names))
                ),
                total_rows=total_rows,
                unified_schema=primary_schema,  # Use first table as "unified"
                merge_strategy="multi_table_join",
                warnings=[],
            )
        else:
            # Single-table mode: original merge behavior
            lazy_frame, merge_info = self._loader.load_sources(
                transformation_input.data_sources
            )
            warnings.extend(merge_info.warnings)

        self._adapter.log_info(
            f"Loaded {merge_info.total_rows:,} rows total"
        )
        
        # 3. Determine LLM mode: BYOK vs Hosted Basic vs Hosted Premium
        self._adapter.log_info("Determining LLM mode...")
        provider, api_key, is_hosted = self._resolve_llm_provider(
            transformation_input
        )
        
        # Determine pricing event based on provider and mode
        pricing_event = self._determine_pricing_event(provider, is_hosted)
        
        mode_label = (
            f"Premium ({provider.display_name})" if pricing_event == PricingEvent.TRANSFORMATION_PREMIUM
            else f"Basic ({provider.display_name})" if is_hosted
            else f"BYOK ({provider.display_name})"
        )
        self._adapter.log_info(f"🔑 Mode: {mode_label}")
        self._adapter.log_info(f"Initializing {provider.display_name}...")
        
        llm_service = LLMService(
            provider=provider,
            api_key=api_key,
            fallback_provider=transformation_input.fallback_provider,
            fallback_api_key=transformation_input.fallback_api_key,
        )
        
        # 4. 🧠 RAG: Search for similar transformations in memory
        similar_codes = []
        rag_storage = self._init_rag_storage()
        
        if rag_storage:
            self._adapter.log_info(
                "🔍 Searching memory for similar transformations..."
            )
            try:
                similar_codes = await rag_storage.search_similar(
                    prompt=transformation_input.prompt,
                    schema=merge_info.unified_schema,
                )
                
                if similar_codes:
                    self._adapter.log_info(
                        f"🧠 Found {len(similar_codes)} similar solutions "
                        f"(similarity > 85%) - reusing knowledge!"
                    )
                    for i, sim in enumerate(similar_codes, 1):
                        self._adapter.log_info(
                            f"  {i}. {sim.score:.1%} similarity "
                            f"(quality: {sim.quality_score:.0f}/100, "
                            f"reused: {sim.reuse_count}x)"
                        )
                else:
                    self._adapter.log_info(
                        "🆕 No similar transformations found - generating fresh code"
                    )
            except Exception as e:
                self._adapter.log_warning(
                    f"RAG search failed: {e}. Continuing without memory."
                )
                similar_codes = []
        else:
            self._adapter.log_info(
                "⚠️ RAG not available - generating without memory"
            )
        
        # 5. Generate transformation code (with RAG context if available)
        self._adapter.log_info("Generating transformation code...")

        generation_result = llm_service.generate_polars_code(
            user_prompt=transformation_input.prompt,
            schema=merge_info.unified_schema,
            similar_codes=similar_codes,  # Pass RAG context to LLM
            max_retries=transformation_input.max_retries,
            multi_table_schemas=multi_table_schemas,  # Pass multi-table schemas if present
        )
        
        self._adapter.log_info(
            f"Code generated in {generation_result.generation_time_ms}ms "
            f"({generation_result.attempts} attempt(s))"
        )
        
        # 6. Execute transformation with error recovery retry
        self._adapter.log_info("Executing transformation...")

        # Prepare dataframes for execution
        if is_multi_table and multi_table_schemas:
            all_dataframes = {
                list(multi_table_schemas.keys())[0]: lazy_frame,
            }
            if extra_dataframes:
                all_dataframes.update(extra_dataframes)
            df_for_single = None
        else:
            all_dataframes = None
            df_for_single = lazy_frame.collect()

        # 6.b Optional oracle generation — one premium LLM call per prompt.
        # The oracle is a property-based contract derived from the natural-
        # language prompt; the deterministic verifier checks the output
        # frame against it (see `executor.oracle`).
        # Enabled by default; disable with `POLARS_RUNNER_DISABLE_ORACLE=1`.
        oracle: Oracle | None = None
        if ORACLE_CONFIG.enabled:
            oracle_key = transformation_input.api_key or self._get_env_api_key(
                LLMProvider.GOOGLE
            )
            if oracle_key:
                try:
                    self._adapter.log_info(
                        f"🧠 Generating PBT oracle ({ORACLE_CONFIG.oracle_model})..."
                    )
                    # Multi-table: pass each table's schema separately so the
                    # oracle sees the *real* column names (no synonyms). For
                    # single-table runs, fall back to the merged schema.
                    oracle_schema: dict | object = (
                        multi_table_schemas if is_multi_table and multi_table_schemas
                        else merge_info.unified_schema
                    )
                    oracle = generate_oracle(
                        prompt=transformation_input.prompt,
                        schema=oracle_schema,
                        llm=_GeminiOracleClient(api_key=oracle_key),
                        model=ORACLE_CONFIG.oracle_model,
                        temperature=ORACLE_CONFIG.oracle_temperature,
                        max_tokens=ORACLE_CONFIG.oracle_max_tokens,
                    )
                    if oracle.is_empty:
                        self._adapter.log_info("  oracle: empty (no claims extractable)")
                    else:
                        self._adapter.log_info(
                            f"  oracle: rows={oracle.expected_rows_range} "
                            f"required_cols={sorted(oracle.required_columns)} "
                            f"non_null={sorted(oracle.non_null_columns)}"
                        )
                except Exception as e:
                    self._adapter.log_warning(
                        f"  oracle generation skipped: {type(e).__name__}: {e}"
                    )
                    oracle = None
            else:
                self._adapter.log_warning(
                    "  oracle enabled but no API key resolved — skipping."
                )

        # Execute with error recovery retry loop
        result_df = await self._execute_with_recovery(
            generation_result=generation_result,
            llm_service=llm_service,
            transformation_input=transformation_input,
            merge_info=merge_info,
            is_multi_table=is_multi_table,
            multi_table_schemas=multi_table_schemas,
            all_dataframes=all_dataframes,
            df_for_single=df_for_single,
            similar_codes=similar_codes,
            rag_storage=rag_storage,
            start_time=start_time,
            oracle=oracle,
        )

        self._adapter.log_info(
            f"Transformation complete: {result_df.height:,} output rows"
        )
        
        # 7. 💾 RAG: Save transformation to memory for future reuse
        if rag_storage:
            # Determine if transformation was truly successful
            # Success = code executed without exception AND produced output rows
            has_output = result_df.height > 0
            
            if has_output:
                self._adapter.log_info(
                    f"💾 Saving SUCCESSFUL transformation to memory "
                    f"({result_df.height} rows produced)..."
                )
            else:
                self._adapter.log_warning(
                    f"⚠️ Saving FAILED transformation to memory "
                    f"(0 rows produced - will not be reused)..."
                )
            
            try:
                await rag_storage.save_transformation(
                    prompt=transformation_input.prompt,
                    code=generation_result.code,
                    schema=merge_info.unified_schema,
                    success=has_output,  # ✅ Success only if produced output rows
                    execution_time_ms=int(
                        (time.perf_counter() - start_time) * 1000
                    ),
                    output_rows=result_df.height,
                    error_message=None,
                    attempts=generation_result.attempts,
                    used_rag=len(similar_codes) > 0,
                    output_columns=result_df.columns,
                    # `_execute_with_recovery` only returns once the semantic
                    # validator returned `ok`. Mark this code as validated so
                    # the retrieval filter prefers it over legacy entries.
                    validator_passed=has_output,
                )
                
                if has_output:
                    self._adapter.log_info("✅ Transformation saved for future reuse")

                    # Increment reuse counter for RAG codes that helped
                    if similar_codes:
                        for sim in similar_codes:
                            try:
                                await rag_storage.increment_reuse_count(sim.id)
                            except Exception:
                                pass  # Non-critical, don't fail the request
                else:
                    self._adapter.log_info(
                        "❌ Transformation saved as FAILED (won't be reused)"
                    )
            except Exception as e:
                self._adapter.log_warning(
                    f"Failed to save to memory: {e}. Continuing anyway."
                )
        
        # 8. Export results
        output_format = transformation_input.output_format
        self._adapter.log_info(
            f"Exporting to {output_format.value}..."
        )
        
        output_bytes = DataExporter.export(result_df, output_format)
        output_filename = DataExporter.get_filename(
            FILE_PATTERNS.output_data,
            output_format,
        )
        content_type = DataExporter.get_content_type(output_format)
        
        await self._adapter.set_value(output_filename, output_bytes, content_type)
        
        # 9. Save generated code if requested
        if transformation_input.include_generated_code:
            await self._adapter.set_value(
                FILE_PATTERNS.generated_code,
                generation_result.code,
            )
        
        # 10. 💰 Charge pricing event (Apify pay-per-event billing)
        await self._adapter.charge_event(
            event_name=pricing_event.value,
            count=1,
            event_data={
                "provider": provider.value,
                "is_hosted": is_hosted,
                "is_premium": pricing_event == PricingEvent.TRANSFORMATION_PREMIUM,
                "tokens_used": generation_result.tokens_used,
                "input_rows": merge_info.total_rows,
                "output_rows": result_df.height,
                "rag_hits": len(similar_codes),  # Number of similar codes found
            }
        )
        
        # 11. Build result
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Get full output data if under size limit
        output_data = None
        try:
            full_data = result_df.to_dicts()
            data_size = len(json.dumps(full_data, default=str))
            if data_size <= LIMITS.max_output_data_bytes:
                output_data = full_data
            else:
                warnings.append(
                    f"Output data too large ({data_size / 1024 / 1024:.1f}MB > 10MB limit). "
                    "Use output_file to download full data."
                )
        except Exception as e:
            warnings.append(f"Could not serialize output_data: {e}")

        result = TransformationResult(
            status=ExecutionStatus.SUCCESS,
            input_sources_count=len(transformation_input.data_sources),
            input_rows_total=merge_info.total_rows,
            input_columns=list(merge_info.unified_schema.columns.keys()),
            output_rows=result_df.height,
            output_columns=result_df.columns,
            output_file=output_filename,
            output_preview=DataExporter.get_preview(result_df),
            output_data=output_data,
            execution_time_ms=execution_time_ms,
            generation_result=generation_result,
            generated_code=generation_result.code if transformation_input.include_generated_code else None,
            warnings=warnings,
            errors=errors,
        )
        
        # 12. Push result to dataset
        await self._adapter.push_data(result.to_dict())
        
        self._adapter.log_info(f"✅ Done! Total time: {execution_time_ms}ms")
        
        return result
    
    def _resolve_llm_provider(
        self, 
        inp: TransformationInput
    ) -> tuple[LLMProvider, str, bool]:
        """
        Resolve which LLM provider to use.
        
        Strategy:
        1. If user provides API key → BYOK mode
           - Google + useAdvancedFeatures=true → Google Pro with grounding
           - Google + useAdvancedFeatures=false → Google basic
           - Groq/others → Use as-is (advanced ignored for non-Google)
        2. If checkbox ON → Premium hosted (Gemini Pro)
        3. If checkbox OFF → Basic hosted (Flash-Lite)
        
        Returns:
            Tuple of (provider, api_key, is_hosted)
        """
        user_api_key = inp.api_key or self._get_env_api_key(inp.llm_provider)
        
        # CASE 1: User provides API key → BYOK mode
        if user_api_key:
            # Sub-case: Google with Advanced Features
            if inp.llm_provider == LLMProvider.GOOGLE and inp.use_advanced_features:
                self._adapter.log_info(
                    "🔑 BYOK mode: Using your Google Pro API with grounding"
                )
                return LLMProvider.GOOGLE_PRO, user_api_key, False
            
            # Standard BYOK (Groq, Google basic, etc)
            return inp.llm_provider, user_api_key, False
        
        # CASE 2: Hosted mode (no user key)
        use_premium = inp.use_advanced_features
        
        if use_premium:
            # Premium hosted: Gemini Pro with reasoning
            hosted_key = os.getenv("GEMINI_PRO_API_KEY")
            if not hosted_key:
                self._adapter.log_warning(
                    "Premium tier not configured. Falling back to basic tier."
                )
                # Fallback to basic
                hosted_key = os.getenv("GEMINI_HOSTED_API_KEY")
                if not hosted_key:
                    raise ValidationError(
                        "No hosted LLM configured. Please provide your own API key."
                    )
                return LLMProvider.GOOGLE_FLASH_LITE, hosted_key, True
            
            self._adapter.log_info(
                "💎 Premium mode: Using Gemini 2.5 Pro with reasoning"
            )
            return LLMProvider.GOOGLE_PRO, hosted_key, True
        else:
            # Basic hosted: Flash-Lite
            hosted_key = os.getenv("GEMINI_HOSTED_API_KEY")
            if not hosted_key:
                raise ValidationError(
                    "No hosted LLM configured. Please provide your own API key."
                )
            self._adapter.log_info(
                "⚡ Basic mode: Using Gemini 2.5 Flash-Lite (hosted)"
            )
            return LLMProvider.GOOGLE_FLASH_LITE, hosted_key, True
    
    def _validate_input(self, inp: TransformationInput) -> None:
        """Validate transformation input."""
        if not inp.prompt:
            raise ValidationError("Prompt is required", field="prompt")
        
        if not inp.data_sources:
            raise ValidationError(
                "At least one data source is required",
                field="data_sources",
            )
        
        if len(inp.prompt) > 10000:
            raise ValidationError(
                f"Prompt too long: {len(inp.prompt)} chars (max 10000)",
                field="prompt",
            )
        
        # Note: We no longer strictly require API key here
        # because we can fall back to hosted mode
    
    def _determine_pricing_event(
        self,
        provider: LLMProvider,
        is_hosted: bool,
    ) -> PricingEvent:
        """
        Determine which pricing event to charge.

        Logic:
        - BYOK: User provided API key → transformation_byok ($0.001)
        - Premium: Using Gemini Pro hosted (our key) → transformation_premium ($0.20)
        - Basic: Using Flash-Lite hosted → transformation_basic ($0.0015)

        Returns:
            PricingEvent enum value
        """
        # BYOK: User provided their own API key (regardless of provider)
        if not is_hosted:
            return PricingEvent.TRANSFORMATION_BYOK

        # Hosted Premium: Using Gemini Pro with our key
        if provider == LLMProvider.GOOGLE_PRO:
            return PricingEvent.TRANSFORMATION_PREMIUM

        # Hosted Basic: Flash-Lite with our key
        return PricingEvent.TRANSFORMATION_BASIC
    
    def _get_env_api_key(self, provider: LLMProvider) -> str | None:
        """Get API key from environment."""
        return os.getenv(provider.env_var_name)

    async def _execute_with_recovery(
        self,
        generation_result,
        llm_service: LLMService,
        transformation_input: TransformationInput,
        merge_info,
        is_multi_table: bool,
        multi_table_schemas: dict | None,
        all_dataframes: dict | None,
        df_for_single: pl.DataFrame | None,
        similar_codes: list,
        rag_storage: TransformationStorage | None,
        start_time: float,
        oracle: Oracle | None = None,
    ) -> pl.DataFrame:
        """
        Execute transformation with error recovery retry.

        If execution fails, regenerates code using error recovery prompt
        and retries up to max_retries times. This allows the LLM to learn
        from errors like missing columns after JOIN operations.

        Args:
            generation_result: Initial code generation result
            llm_service: LLM service for regeneration
            transformation_input: Original transformation input
            merge_info: Merged dataset info
            is_multi_table: Whether this is multi-table mode
            multi_table_schemas: Schema info for multi-table mode
            all_dataframes: Dataframes dict for multi-table mode
            df_for_single: DataFrame for single-table mode
            similar_codes: RAG context codes
            rag_storage: RAG storage for saving failures
            start_time: Pipeline start time for timing

        Returns:
            Transformed DataFrame

        Raises:
            Exception: If all retry attempts fail
        """
        current_code = generation_result.code
        total_attempts = generation_result.attempts
        last_error: str | None = None

        # Try execution with up to max_retries recovery attempts
        for recovery_attempt in range(transformation_input.max_retries):
            try:
                if is_multi_table and all_dataframes:
                    result_df = execute_multi_table_transformation(
                        code=current_code,
                        dataframes=all_dataframes,
                    )
                else:
                    result_df = execute_transformation(
                        code=current_code,
                        df=df_for_single,
                    )

                # Execution did not raise — run the semantic validator before
                # declaring victory. A `fail` verdict is treated as a retry-able
                # error so the next attempt can fix the wrong group key / null
                # propagation / etc.
                verdict = validate_semantic(
                    code=current_code,
                    result_df=result_df,
                )
                if verdict.is_fail:
                    self._adapter.log_warning(
                        f"🩺 Semantic validator [{verdict.layer}] flagged the output: "
                        f"{'; '.join(verdict.reasons)[:200]}"
                    )
                    raise SemanticValidationFailure(verdict)
                if verdict.is_warn:
                    self._adapter.log_info(
                        f"⚠️ Semantic validator [{verdict.layer}] warning (kept result): "
                        f"{'; '.join(verdict.reasons)[:200]}"
                    )

                # Oracle gate — checks prompt-derived properties (rows
                # cardinality, required columns, non-null constraints, value
                # ranges, monotonicity). Domain- and language-agnostic by
                # construction. Only runs when ORACLE_CONFIG.enabled and the
                # generator produced a non-empty oracle.
                if oracle is not None and not oracle.is_empty:
                    oracle_verdict = verify_against_oracle(oracle, result_df)
                    if not oracle_verdict.passed and ORACLE_CONFIG.oracle_is_blocking:
                        self._adapter.log_warning(
                            f"🧪 Oracle gate FAIL (score {oracle_verdict.score:.0f}/100): "
                            f"{oracle_verdict.feedback_hint[:240]}"
                        )
                        raise SemanticValidationFailure(
                            SemanticVerdict(
                                level=VerdictLevel.FAIL,
                                reasons=list(oracle_verdict.failed_rules),
                                layer="deterministic",
                                suggested_fix=oracle_verdict.feedback_hint,
                            )
                        )
                    if oracle_verdict.passed:
                        self._adapter.log_info(
                            f"🧪 Oracle gate PASS (score {oracle_verdict.score:.0f}/100)"
                        )

                # Success! Return result
                if recovery_attempt > 0:
                    self._adapter.log_info(
                        f"✅ Error recovery succeeded after {recovery_attempt} retry(s)"
                    )
                return result_df

            except Exception as exec_error:
                last_error = str(exec_error)
                total_attempts += 1

                # Analyze the error for intelligent recovery
                error_analysis = ErrorAnalyzer.analyze(exec_error, current_code)

                # Log the error with analysis
                self._adapter.log_warning(
                    f"⚠️ Execution failed (attempt {recovery_attempt + 1}/"
                    f"{transformation_input.max_retries}): [{error_analysis.error_type.value}] "
                    f"{error_analysis.message[:150]}"
                )

                # Save failure to RAG for learning
                if rag_storage:
                    try:
                        await rag_storage.save_transformation(
                            prompt=transformation_input.prompt,
                            code=current_code,
                            schema=merge_info.unified_schema,
                            success=False,
                            execution_time_ms=int(
                                (time.perf_counter() - start_time) * 1000
                            ),
                            output_rows=0,
                            error_message=last_error,
                            attempts=total_attempts,
                            used_rag=len(similar_codes) > 0,
                            output_columns=None,
                        )
                    except Exception:
                        pass  # Non-critical

                # Check if we have more retries
                if recovery_attempt >= transformation_input.max_retries - 1:
                    # No more retries, raise the error
                    raise

                # Try to regenerate code with intelligent error recovery
                self._adapter.log_info(
                    f"🔄 Attempting intelligent error recovery ({error_analysis.fix_hint})..."
                )

                try:
                    recovery_result = llm_service.generate_polars_code(
                        user_prompt=transformation_input.prompt,
                        schema=merge_info.unified_schema,
                        similar_codes=similar_codes,
                        max_retries=1,  # Single attempt for recovery
                        multi_table_schemas=multi_table_schemas,
                        previous_code=current_code,  # Pass failed code
                        previous_error=last_error,  # Pass error message
                        error_analysis=error_analysis,  # Pass structured analysis!
                    )

                    # Update code for next iteration
                    current_code = recovery_result.code
                    total_attempts += recovery_result.attempts

                    self._adapter.log_info(
                        f"🔧 New code generated with fix for {error_analysis.error_type.value}, retrying..."
                    )
                except Exception as regen_error:
                    self._adapter.log_warning(
                        f"⚠️ Code regeneration failed: {regen_error}"
                    )
                    # Continue to next iteration or raise original error
                    if recovery_attempt >= transformation_input.max_retries - 1:
                        raise exec_error from regen_error

        # Should not reach here, but just in case
        raise RuntimeError(f"Execution failed after {total_attempts} attempts: {last_error}")


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main() -> None:
    """Main entry point."""
    async with ApifyAdapter() as adapter:
        try:
            # Get input
            raw_input = await adapter.get_input()
            
            if not raw_input:
                adapter.log_error("No input provided")
                return
            
            adapter.log_info("🚀 Starting AI Data Transformer...")
            
            # Run transformation
            transformer = Transformer(adapter)
            await transformer.run(raw_input)
            
        except TransformerError as e:
            adapter.log_error(f"❌ Transformation failed: {e.user_message}")
            await adapter.push_data({
                "status": ExecutionStatus.FAILED.value,
                "error": e.to_dict(),
            })
            raise
            
        except Exception as e:
            adapter.log_error(f"❌ Unexpected error: {e}")
            await adapter.push_data({
                "status": ExecutionStatus.FAILED.value,
                "error": {"message": str(e), "type": type(e).__name__},
            })
            raise


def cli() -> None:
    """Sync entry point for the `polars-runner` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
