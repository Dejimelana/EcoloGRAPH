"""
Entity extractor using LLM for ecological data extraction.

Extracts species, measurements, locations, and relationships from
document chunks with full traceability to source text.
"""
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime

from ..core.llm_client import LLMClient
from ..core.schemas import (
    SourceReference,
    SpeciesMention,
    Measurement,
    Location,
    TemporalRange,
    EcologicalRelation,
    RelationType,
    ExtractionResult,
    Coordinates,
    GenericEntity,
    Method,
    Dataset,
)
from ..ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts ecological entities from document chunks using LLM.
    
    Features:
    - Structured extraction with Pydantic validation
    - Source traceability for each extraction
    - Batch processing with progress tracking
    - Configurable prompts
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompts_dir: Path | str | None = None
    ):
        """
        Initialize entity extractor.
        
        Args:
            llm_client: LLM client instance (creates default if None)
            prompts_dir: Directory containing prompt templates
        """
        self.llm = llm_client or LLMClient(role="ingestion")
        
        # Load prompts
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            self.prompts_dir = Path(__file__).parent.parent.parent / "config" / "prompts"
        
        
        self.system_prompt = self._load_prompt("extraction_system.txt")
        self.user_prompt_template = self._load_prompt("extraction_user.txt")
        self.paper_prompt_template = self._load_prompt("extraction_paper.txt")
        self.generic_prompt_template = self._load_prompt("extraction_generic.txt")  # NEW: fallback prompt
        
        logger.info("EntityExtractor initialized")
    
    def extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        """
        Extract entities from a single document chunk.
        
        Args:
            chunk: Document chunk to process
            
        Returns:
            ExtractionResult with extracted entities
        """
        # Build prompt
        user_prompt = self.user_prompt_template.format(
            text=chunk.text,
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            section=chunk.section or "Unknown"
        )
        
        # Call LLM
        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            # Parse response
            raw_data = self._parse_response(response.content)
            
            # Convert to typed entities
            result = self._build_extraction_result(
                raw_data=raw_data,
                chunk=chunk,
                model_used=self.llm.model
            )
            
            logger.debug(f"Extracted {result.entity_count()} entities from chunk {chunk.chunk_id}")
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk.chunk_id}: {e}")
            return ExtractionResult(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                model_used=self.llm.model,
                extraction_confidence=0.0
            )
    
    def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
        progress_callback=None,
        enable_auto_restart: bool = True
    ) -> list[ExtractionResult]:
        """
        Extract entities from multiple chunks with auto-restart on LLM failure.
        
        Args:
            chunks: List of chunks to process
            progress_callback: Optional callback(current, total) for progress
            enable_auto_restart: Enable automatic LM Studio restart on failure
            
        Returns:
            List of ExtractionResult for each chunk
        """
        results = []
        total = len(chunks)
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        # Initialize LM Studio manager if auto-restart enabled
        lm_studio_manager = None
        if enable_auto_restart:
            try:
                from ..core.lm_studio_manager import LMStudioManager
                lm_studio_manager = LMStudioManager(
                    base_url=self.llm.base_url if hasattr(self.llm, 'base_url') else "http://localhost:1234"
                )
                logger.info("LM Studio auto-restart enabled")
            except Exception as e:
                logger.warning(f"Could not initialize LMStudioManager: {e}")
                enable_auto_restart = False
        
        for i, chunk in enumerate(chunks):
            result = self.extract_from_chunk(chunk)
            results.append(result)
            
            # Track failures for auto-restart
            if result.entity_count() == 0 and result.extraction_confidence == 0.0:
                consecutive_failures += 1
                logger.warning(
                    f"Extraction failed for chunk {chunk.chunk_id} "
                    f"({consecutive_failures}/{max_consecutive_failures} consecutive failures)"
                )
                
                # Trigger restart if threshold reached
                if consecutive_failures >= max_consecutive_failures and enable_auto_restart and lm_studio_manager:
                    logger.error(
                        f"Detected {consecutive_failures} consecutive failures, "
                        "LM Studio may be corrupted. Attempting restart..."
                    )
                    
                    if lm_studio_manager.restart():
                        logger.info("LM Studio restarted successfully, resuming extraction")
                        consecutive_failures = 0  # Reset counter
                    else:
                        logger.error(
                            "LM Studio restart failed. Continuing with potential errors. "
                            "You may need to manually restart LM Studio."
                        )
                        # Don't reset counter, but continue trying
            else:
                # Success, reset failure counter
                consecutive_failures = 0
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        total_entities = sum(r.entity_count() for r in results)
        logger.info(f"Extracted {total_entities} entities from {total} chunks")
        
        return results
    
    def extract_from_paper(
        self,
        chunks: list[DocumentChunk],
        paper_metadata: dict | None = None,
        context_window: int = 2048,
        max_output_tokens: int = 500
    ) -> ExtractionResult:
        """
        Extract entities from entire paper at once (paper as functional unit).
        
        Strategy:
        1. Try to process all chunks together in one LLM call
        2. If exceeds context window, automatically batch into multiple calls
        3. Consolidate results from batches
        
        This is the RECOMMENDED method for entity extraction as it:
        - Preserves cross-chunk context
        - Resolves entity cross-references ("the species" → actual name)
        - Deduplicates entities across chunks
        - Validates context before calling LLM
        
        Args:
            chunks: All chunks from the paper
            paper_metadata: Optional dict with title, authors, year
            context_window: Model's context window (tokens)
            max_output_tokens: Reserve for model output
            
        Returns:
            Single ExtractionResult for entire paper
        """
        from ..core.token_utils import estimate_tokens, estimate_safe_batch_size
        
        if not chunks:
            logger.warning("extract_from_paper called with empty chunks")
            return ExtractionResult(
                doc_id="unknown",
                model_used=self.llm.model,
                extraction_confidence=0.0
            )
        
        doc_id = chunks[0].doc_id
        metadata = paper_metadata or {}
        
        # Build full paper text
        full_paper_text = self._format_chunks_for_paper_extraction(chunks, metadata)
        
        # Estimate tokens
        system_tokens = estimate_tokens(self.system_prompt)
        paper_tokens = estimate_tokens(full_paper_text)
        total_input = system_tokens + paper_tokens
        total_request = total_input + max_output_tokens
        
        logger.info(
            f"Paper {doc_id}: {len(chunks)} chunks, "
            f"~{total_request} tokens (input: {total_input}, output: {max_output_tokens})"
        )
        
        # Strategy 1: Try entire paper first
        if total_request <= context_window:
            logger.info(f"Processing entire paper in one call (fits in context)")
            result = self._extract_from_full_paper(chunks, metadata)
            
            # NEW: Generic fallback for single-call extraction too
            ecological_count = (
                len(result.species) +
                len(result.measurements) +
                len(result.locations) +
                len(result.relations)
            )
            
            if ecological_count < 5 and hasattr(self, 'generic_prompt_template'):
                logger.info(
                    f"Low ecological yield ({ecological_count} entities), "
                    f"trying generic extraction as fallback..."
                )
                try:
                    original_template = self.paper_prompt_template
                    self.paper_prompt_template = self.generic_prompt_template
                    
                    generic_result = self._extract_from_full_paper(chunks, metadata)
                    
                    self.paper_prompt_template = original_template
                    
                    if generic_result.entity_count() > result.entity_count():
                        logger.info(
                            f"Generic extraction yielded more entities "
                            f"({generic_result.entity_count()} vs {result.entity_count()}), using it"
                        )
                        return generic_result
                except Exception as e:
                    logger.warning(f"Generic fallback failed: {e}")
            
            return result
        
        # Strategy 2: Paper too large, batch automatically
        logger.warning(
            f"Paper exceeds context ({total_request} > {context_window}), "
            f"using automatic batching"
        )
        result = self._extract_from_paper_batched(
            chunks, 
            metadata, 
            context_window, 
            max_output_tokens
        )
        
        # NEW: Generic fallback if few ecological entities found
        ecological_count = (
            len(result.species) +
            len(result.measurements) +
            len(result.locations) +
            len(result.relations)
        )
        
        if ecological_count < 5 and hasattr(self, 'generic_prompt_template'):
            logger.info(
                f"Low ecological yield ({ecological_count} entities), "
                f"trying generic extraction as fallback..."
            )
            try:
                # Replace paper_prompt_template temporarily
                original_template = self.paper_prompt_template
                self.paper_prompt_template = self.generic_prompt_template
                
                # Re-run extraction with generic prompt
                generic_result = self._extract_from_paper_batched(
                    chunks,
                    metadata,
                    context_window,
                    max_output_tokens
                )
                
                # Restore original template
                self.paper_prompt_template = original_template
                
                # Use generic result if it found more entities
                if generic_result.entity_count() > result.entity_count():
                    logger.info(
                        f"Generic extraction yielded more entities "
                        f"({generic_result.entity_count()} vs {result.entity_count()}), using it"
                    )
                    return generic_result
            except Exception as e:
                logger.warning(f"Generic fallback failed: {e}")
        
        return result
    
    def _format_chunks_for_paper_extraction(
        self,
        chunks: list[DocumentChunk],
        metadata: dict
    ) -> str:
        """Format all chunks into paper extraction prompt."""
        # Build chunks section
        chunks_text = []
        for i, chunk in enumerate(chunks):
            chunk_header = f"\n---\nCHUNK {i+1}"
            if chunk.page:
                chunk_header += f" (Page {chunk.page}"
            if chunk.section:
                chunk_header += f", Section: {chunk.section}"
            if chunk.page or chunk.section:
                chunk_header += ")"
            chunk_header += ":\n"
            
            chunks_text.append(chunk_header + chunk.text)
        
        # Format full prompt
        prompt = self.paper_prompt_template.format(
            title=metadata.get('title', 'Unknown'),
            authors=', '.join(metadata.get('authors', [])) or 'Unknown',
            year=metadata.get('year', 'Unknown'),
            chunks='\n'.join(chunks_text)
        )
        
        return prompt
    
    def _extract_from_full_paper(
        self,
        chunks: list[DocumentChunk],
        metadata: dict
    ) -> ExtractionResult:
        """Extract from entire paper in one LLM call."""
        try:
            # Build prompt
            user_prompt = self._format_chunks_for_paper_extraction(chunks, metadata)
            
            # Call LLM
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,
                max_tokens=2048
            )
            
            # Debug: log raw response for diagnosis
            raw_content = response.content
            logger.debug(f"Raw LLM response (first 500 chars): {raw_content[:500]}")
            if '<think>' in raw_content:
                logger.info(f"Thinking tokens detected, stripping before parse")
            
            # Parse response
            raw_data = self._parse_response(raw_content)
            
            # Build result (paper-level)
            result = self._build_extraction_result(
                raw_data=raw_data,
                chunk=chunks[0],  # Use first chunk for doc_id
                model_used=self.llm.model
            )
            
            # Override doc_id to be paper-level
            result.doc_id = chunks[0].doc_id
            result.chunk_id = f"{chunks[0].doc_id}_paper"  # Indicate paper-level
            
            logger.info(
                f"Extracted {result.entity_count()} entities from paper "
                f"(1 LLM call, {len(chunks)} chunks)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Full paper extraction failed: {e}")
            return ExtractionResult(
                doc_id=chunks[0].doc_id,
                chunk_id=f"{chunks[0].doc_id}_paper",
                model_used=self.llm.model,
                extraction_confidence=0.0
            )
    
    def _extract_from_paper_batched(
        self,
        chunks: list[DocumentChunk],
        metadata: dict,
        context_window: int,
        max_output_tokens: int
    ) -> ExtractionResult:
        """Extract from paper using automatic batching."""
        from ..core.token_utils import estimate_safe_batch_size
        
        # Calculate batch size
        chunk_texts = [chunk.text for chunk in chunks]
        batch_size = estimate_safe_batch_size(
            chunk_texts=chunk_texts,
            system_prompt=self.system_prompt,
            max_output_tokens=max_output_tokens,
            context_window=context_window,
            template_overhead=300  # For chunk formatting
        )
        
        logger.info(
            f"Batching {len(chunks)} chunks into groups of ~{batch_size} "
            f"(estimated safe batch size)"
        )
        
        # Create batches
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batches.append(batch_chunks)
        
        logger.info(f"Created {len(batches)} batches")
        
        # Extract from each batch
        batch_results = []
        for i, batch_chunks in enumerate(batches):
            logger.info(
                f"Processing batch {i+1}/{len(batches)} "
                f"({len(batch_chunks)} chunks)"
            )
            
            try:
                batch_result = self._extract_from_full_paper(
                    batch_chunks,
                    metadata
                )
                batch_results.append(batch_result)
                
            except Exception as e:
                logger.error(f"Batch {i+1} extraction failed: {e}")
                # Continue with other batches
        
        # Consolidate results
        if not batch_results:
            logger.error("All batches failed")
            return ExtractionResult(
                doc_id=chunks[0].doc_id,
                chunk_id=f"{chunks[0].doc_id}_paper",
                model_used=self.llm.model,
                extraction_confidence=0.0
            )
        
        # Merge results (simple concatenation for now)
        merged = ExtractionResult(
            doc_id=chunks[0].doc_id,
            chunk_id=f"{chunks[0].doc_id}_paper",
            model_used=self.llm.model,
            extraction_confidence=sum(r.extraction_confidence for r in batch_results) / len(batch_results)
        )
        
        for result in batch_results:
            merged.species.extend(result.species)
            merged.measurements.extend(result.measurements)
            merged.locations.extend(result.locations)
            merged.temporal_info.extend(result.temporal_info)
            merged.relations.extend(result.relations)
        
        logger.info(
            f"Merged {len(batch_results)} batches → "
            f"{merged.entity_count()} total entities "
            f"({len(batches)} LLM calls)"
        )
        
        return merged
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        path = self.prompts_dir / filename
        
        if path.exists():
            return path.read_text(encoding="utf-8")
        else:
            logger.warning(f"Prompt file not found: {path}")
            return ""
    
    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response to JSON."""
        import re
        # Clean response
        content = content.strip()
        
        # Remove Qwen3 thinking tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        # Find JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        
        if start == -1 or end <= start:
            logger.warning("No JSON object found in response")
            return {}
        
        json_str = content[start:end]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find the correct closing brace by counting nesting
            depth = 0
            for i, ch in enumerate(content[start:], start=start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = content[start:i+1]
                        break
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                logger.debug(f"Raw JSON (first 300 chars): {json_str[:300]}")
                return {}
    
    def _build_extraction_result(
        self,
        raw_data: dict[str, Any],
        chunk: DocumentChunk,
        model_used: str
    ) -> ExtractionResult:
        """Build ExtractionResult from raw LLM output."""
        result = ExtractionResult(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            model_used=model_used,
            extraction_timestamp=datetime.now(),
            extraction_confidence=1.0 if raw_data else 0.0
        )
        
        # Parse species
        for item in raw_data.get("species", []):
            try:
                species = SpeciesMention(
                    original_name=item.get("original_name", ""),
                    common_name=item.get("common_name"),
                    source=self._create_source_ref(
                        chunk=chunk,
                        text_snippet=item.get("text_snippet")
                    )
                )
                if species.original_name:
                    result.species.append(species)
            except Exception as e:
                logger.debug(f"Failed to parse species: {e}")
        
        # Parse measurements
        for item in raw_data.get("measurements", []):
            try:
                measurement = Measurement(
                    parameter=item.get("parameter", ""),
                    value=float(item.get("value", 0)),
                    unit=item.get("unit", ""),
                    value_min=item.get("value_min"),
                    value_max=item.get("value_max"),
                    species=item.get("species"),
                    life_stage=item.get("life_stage"),
                    is_mean=item.get("is_mean", False),
                    source=self._create_source_ref(
                        chunk=chunk,
                        text_snippet=item.get("text_snippet")
                    )
                )
                if measurement.parameter and measurement.value:
                    result.measurements.append(measurement)
            except Exception as e:
                logger.debug(f"Failed to parse measurement: {e}")
        
        # Parse locations
        for item in raw_data.get("locations", []):
            try:
                coords = None
                if item.get("latitude") and item.get("longitude"):
                    coords = Coordinates(
                        latitude=float(item["latitude"]),
                        longitude=float(item["longitude"])
                    )
                
                location = Location(
                    name=item.get("name", ""),
                    coordinates=coords,
                    region=item.get("region"),
                    country=item.get("country"),
                    habitat_type=item.get("habitat_type"),
                    source=self._create_source_ref(
                        chunk=chunk,
                        text_snippet=item.get("text_snippet")
                    )
                )
                if location.name:
                    result.locations.append(location)
            except Exception as e:
                logger.debug(f"Failed to parse location: {e}")
        
        # Parse temporal info
        for item in raw_data.get("temporal", []):
            try:
                temporal = TemporalRange(
                    start_date=item.get("start_date"),
                    end_date=item.get("end_date"),
                    season=item.get("season"),
                    description=item.get("description"),
                    source=self._create_source_ref(
                        chunk=chunk,
                        text_snippet=item.get("text_snippet")
                    )
                )
                result.temporal_info.append(temporal)
            except Exception as e:
                logger.debug(f"Failed to parse temporal: {e}")
        
        # Parse relations
        for item in raw_data.get("relations", []):
            try:
                rel_type = item.get("relation_type", "associated")
                try:
                    rel_type_enum = RelationType(rel_type)
                except ValueError:
                    rel_type_enum = RelationType.ASSOCIATED_WITH
                
                relation = EcologicalRelation(
                    relation_type=rel_type_enum,
                    subject=item.get("subject", ""),
                    object=item.get("object", ""),
                    description=item.get("description"),
                    source=self._create_source_ref(
                        chunk=chunk,
                        text_snippet=item.get("text_snippet")
                    )
                )
                if relation.subject and relation.object:
                    result.relations.append(relation)
            except Exception as e:
                logger.debug(f"Failed to parse relation: {e}")
        
        return result
    
    def _create_source_ref(
        self,
        chunk: DocumentChunk,
        text_snippet: str | None = None
    ) -> SourceReference:
        """Create source reference from chunk."""
        return SourceReference(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            page=chunk.page,
            section=chunk.section,
            text_snippet=text_snippet,
            confidence=1.0,
            is_inferred=False
        )
