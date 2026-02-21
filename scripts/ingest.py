"""
EcoloGRAPH Ingestion Pipeline.

End-to-end pipeline: PDF → Parse → Chunk → Classify → Extract → Index
Populates all three stores:
  - SQLite FTS5 (PaperIndex) for keyword search
  - Qdrant (VectorStore) for semantic search
  - Neo4j (GraphBuilder) for knowledge graph

Usage:
    python scripts/ingest.py data/raw/          # Process all PDFs in directory
    python scripts/ingest.py paper.pdf           # Process single PDF
    python scripts/ingest.py data/raw/ --skip-graph   # Skip Neo4j (if not running)
    python scripts/ingest.py data/raw/ --skip-extract # Skip entity extraction (no LLM needed)
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.extraction.domain_classifier import DomainClassifier
from src.search.paper_index import PaperIndex, IndexedPaper
from src.core.config import _load_api_key_file
from src.core.lm_studio_manager import LMStudioManager

# Load API key from config/api-key before any LLM calls
_load_api_key_file()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ingest")


def create_indexed_paper(doc, classification) -> IndexedPaper:
    """Create an IndexedPaper from parsed document and classification."""
    import re
    
    title = doc.title
    year = None
    
    # Fallback: extract title and year from filename if parser didn't get them
    if not title or title == "Untitled":
        filename = Path(doc.source_path).stem if doc.source_path else ""
        if filename:
            # ScienceDirect pattern: Title-words_YEAR_Journal.pdf
            year_match = re.search(r'_(\d{4})_', filename)
            if year_match:
                year = int(year_match.group(1))
                title = filename[:year_match.start()]
            else:
                title = filename
            
            # Clean up: hyphens → spaces, normalize
            title = re.sub(r'---+', ' — ', title)
            title = re.sub(r'--', ' – ', title)
            title = title.replace('-', ' ').replace('_', ' ')
            title = re.sub(r'\s+', ' ', title).strip()
            if title:
                title = title[0].upper() + title[1:]
            # Remove trailing single chars (filename truncation artifacts)
            title = re.sub(r'\s+\w$', '', title)
    
    if not title:
        title = "Untitled"
    
    # Also try to get year from parsed text if not from filename
    if not year and hasattr(doc, '_extracted_year'):
        year = doc._extracted_year
    
    return IndexedPaper(
        doc_id=doc.doc_id,
        title=title,
        authors=doc.authors or [],
        year=year,
        journal=None,
        abstract=doc.abstract,
        keywords=[],
        primary_domain=classification.primary_domain.value if classification else None,
        domains={
            dt.value: score 
            for dt, score in (classification.domain_scores.items() if classification else {})
        },
        study_type=classification.study_type.value if classification else None,
        source_path=str(Path(doc.source_path).resolve()) if doc.source_path else None,
        indexed_at=datetime.now(),
        doi=getattr(doc, 'doi', None),
    )


def enhance_metadata_with_pymupdf(doc) -> None:
    """
    Enhance parsed document metadata using PyMuPDF font-size analysis.
    
    This runs as a fallback when Docling fails to extract title/abstract.
    PyMuPDF is more reliable for metadata extraction because it analyzes
    font sizes (largest font = title) rather than text patterns.
    
    Modifies doc in-place.
    """
    import re
    from collections import defaultdict
    
    try:
        import pymupdf
    except ImportError:
        logger.warning("PyMuPDF not available for metadata enhancement")
        return
    
    # Only enhance if metadata is missing or looks wrong
    needs_title = (
        not doc.title or 
        doc.title == "Untitled" or 
        len(doc.title) < 5 or
        "<!--" in doc.title or  # Docling error
        "image" in doc.title.lower()
    )
    needs_abstract = not doc.abstract or len(doc.abstract or "") < 30
    needs_year = not doc.year
    
    if not needs_title and not needs_abstract and not needs_year:
        return  # Nothing to enhance
    
    pdf_path = Path(doc.source_path)
    if not pdf_path.exists():
        logger.debug(f"PDF not found for metadata enhancement: {pdf_path}")
        return
    
    try:
        pdf_doc = pymupdf.open(str(pdf_path))
        if pdf_doc.page_count == 0:
            pdf_doc.close()
            return
        
        page = pdf_doc[0]
        
        # --- TITLE: Find largest font text on page 1 ---
        if needs_title:
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
            all_spans = []
            
            for block in blocks:
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        if text and size > 0:
                            all_spans.append((size, text))
            
            if all_spans:
                # Find the largest font size (title)
                max_size = max(s for s, t in all_spans if len(t) > 3)
                
                # Collect all text at that size (title may span multiple lines)
                title_parts = []
                for size, text in all_spans:
                    if abs(size - max_size) < 0.5 and len(text) > 2:
                        # Skip purely numeric fragments
                        if not re.match(r'^[\d\s.,;:]+$', text):
                            title_parts.append(text)
                
                if title_parts:
                    title = " ".join(title_parts).strip()
                    title = re.sub(r'\s+', ' ', title)
                    title = title.strip('.,;: ')
                    if 10 < len(title) < 500:
                        doc.title = title
                        logger.info(f"  ✅ Enhanced title via PyMuPDF: {title[:60]}...")
        
        # --- ABSTRACT: Extract from first pages ---
        if needs_abstract:
            pages_text = ""
            for i in range(min(3, pdf_doc.page_count)):
                pages_text += pdf_doc[i].get_text() + "\n"
            
            # Strategy 1: "Abstract" heading
            abs_match = re.search(
                r'[Aa]bstract\.?\s*\n?\s*(.*?)(?=\s*(?:Keywords?|Introduction|INTRODUCTION|1\.\s|©|\bDOI\b))',
                pages_text, re.DOTALL
            )
            if abs_match:
                abstract = abs_match.group(1).strip()
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                if 50 < len(abstract) < 3000:
                    doc.abstract = abstract
                    logger.info(f"  ✅ Enhanced abstract via PyMuPDF ({len(abstract)} chars)")
        
        # --- YEAR: Extract from PDF metadata or first page ---
        if needs_year:
            # Try PDF metadata first
            metadata_year = None
            if pdf_doc.metadata.get("creationDate"):
                # Format: "D:20240315..." -> 2024
                date_str = pdf_doc.metadata["creationDate"]
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    metadata_year = int(year_match.group(1))
            
            # Try first page text patterns
            first_page_text = pdf_doc[0].get_text()
            
            # Common patterns: (2024), ©2024, 2024, "Published: 2024"
            year_patterns = [
                r'\((\d{4})\)',  # (2024)
                r'©\s*(\d{4})',  # ©2024
                r'[Pp]ublished[:\s]+(\d{4})',  # Published: 2024
                r'\b(20\d{2})\b',  # Any year 2000-2099
            ]
            
            text_year = None
            for pattern in year_patterns:
                match = re.search(pattern, first_page_text)
                if match:
                    year_candidate = int(match.group(1))
                    # Sanity check: reasonable year range
                    if 1990 <= year_candidate <= 2030:
                        text_year = year_candidate
                        break
            
            # Prefer text year over metadata year (more accurate)
            doc.year = text_year or metadata_year
            if doc.year:
                logger.info(f"  ✅ Enhanced year via PyMuPDF: {doc.year}")
        
        pdf_doc.close()
        
    except Exception as e:
        logger.debug(f"PyMuPDF metadata enhancement failed: {e}")



def ingest(
    input_path: Path,
    skip_extract: bool = False,
    skip_graph: bool = False,
    skip_vectors: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    restart_interval: int = 0,
    model: str | None = None,
    thinking: bool = False,
    max_tokens: int = 2048,
    timeout: float = 120.0,
):
    """Run the full ingestion pipeline."""
    
    # Collect PDF paths
    if input_path.is_file():
        pdf_paths = [input_path]
    elif input_path.is_dir():
        pdf_paths = sorted(input_path.glob("**/*.pdf"))
    else:
        logger.error(f"Path not found: {input_path}")
        return
    
    if not pdf_paths:
        logger.warning(f"No PDF files found in {input_path}")
        return
    
    logger.info(f"📄 Found {len(pdf_paths)} PDF(s) to process")
    
    # Initialize components
    parser = PDFParser()
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        respect_sections=True,
    )
    classifier = DomainClassifier()
    paper_index = PaperIndex()
    
    # Optional components
    vector_store = None
    graph_builder = None
    entity_extractor = None
    
    if not skip_vectors:
        try:
            from src.retrieval.vector_store import VectorStore
            vector_store = VectorStore()
            logger.info("✅ Qdrant vector store connected")
        except Exception as e:
            logger.warning(f"⚠️  Qdrant not available, skipping vectors: {e}")
    
    if not skip_graph:
        try:
            from src.graph.graph_builder import GraphBuilder
            graph_builder = GraphBuilder()
            graph_builder.initialize_schema()
            logger.info("✅ Neo4j graph database connected")
        except Exception as e:
            logger.warning(f"⚠️  Neo4j not available, skipping graph: {e}")
    
    if not skip_extract:
        try:
            from src.extraction.entity_extractor import EntityExtractor
            from src.extraction.citation_extractor import CitationExtractor
            from src.core.llm_client import LLMClient
            
            # Build LLM client with user-specified parameters
            llm_kwargs = {"role": "ingestion", "timeout": timeout}
            if model:
                llm_kwargs["model"] = model
            llm_kwargs["max_tokens"] = max_tokens
            llm_client = LLMClient(**llm_kwargs)
            
            entity_extractor = EntityExtractor(llm_client=llm_client)
            citation_extractor = CitationExtractor(llm_client=llm_client)
            
            # Configure thinking mode
            if not thinking:
                logger.info("🧠 Thinking mode: OFF (fast extraction, /no_think)")
            else:
                logger.info("🧠 Thinking mode: ON (slower but may improve quality)")
                # Remove /no_think from system prompt if thinking is enabled
                if hasattr(entity_extractor, 'system_prompt') and '/no_think' in entity_extractor.system_prompt:
                    entity_extractor.system_prompt = entity_extractor.system_prompt.replace('/no_think', '').strip()
            
            logger.info(
                f"✅ Extractors initialized: model={llm_client.model}, "
                f"max_tokens={max_tokens}, timeout={timeout}s"
            )
        except Exception as e:
            logger.warning(f"⚠️  Entity extractor not available: {e}")
    
    # Initialize LM Studio manager for periodic restarts
    lm_studio_manager = None
    if not skip_extract and restart_interval > 0:
        try:
            lm_studio_manager = LMStudioManager()
            logger.info(f"✓ LM Studio manager initialized (restart every {restart_interval} papers)")
        except Exception as e:
            logger.warning(f"Could not initialize LM Studio manager: {e}")
            logger.warning(f"Periodic restarts disabled")
    
    # Process each PDF
    total_chunks = 0
    processed = 0
    failed = 0
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        # Periodic LM Studio restart to prevent memory leaks
        if lm_studio_manager and restart_interval > 0 and i > 1 and (i - 1) % restart_interval == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 RESTARTING LM STUDIO (processed {i-1} papers)")
            logger.info(f"   Reason: Prevent memory leak accumulation")
            logger.info(f"{'='*60}")
            
            try:
                if lm_studio_manager.restart():
                    logger.info("✅ LM Studio restarted successfully")
                else:
                    logger.warning("⚠️  LM Studio restart failed, continuing anyway...")
            except Exception as e:
                logger.error(f"❌ LM Studio restart error: {e}")
                logger.warning("Continuing without restart...")
        
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(pdf_paths)}] Processing: {pdf_path.name}")
        
        try:
            # Step 1: Parse PDF
            doc = parser.parse(pdf_path)
            logger.info(f"  📖 Parsed: {doc.num_pages} pages, {len(doc.sections)} sections, {len(doc.tables)} tables")
            
            # Step 1.5: Enhance metadata with PyMuPDF if needed
            enhance_metadata_with_pymupdf(doc)
            if doc.title:
                logger.info(f"  📝 Title: {doc.title[:60]}{'...' if len(doc.title) > 60 else ''}")
            if doc.abstract:
                logger.info(f"  📄 Abstract: {len(doc.abstract)} chars")
            
            # Step 2: Classify document
            classification = classifier.classify_document(doc, use_llm=False)
            domain = classification.primary_domain.value
            confidence = classification.confidence
            logger.info(f"  🏷️  Domain: {domain} ({confidence:.0%})")
            
            # Step 3: Chunk document
            chunks = chunker.chunk_document(doc)
            total_chunks += len(chunks)
            logger.info(f"  📦 Chunks: {len(chunks)}")
            
            # Step 4: Index in PaperIndex (SQLite)
            indexed = create_indexed_paper(doc, classification)
            paper_index.add_paper(indexed)
            logger.info(f"  💾 Indexed in SQLite")
            
            # Step 4b: Store chunks in SQLite (always available, no Qdrant needed)
            try:
                paper_index.add_chunks(chunks)
                logger.info(f"  📦 {len(chunks)} chunks stored in SQLite")
            except Exception as e:
                logger.warning(f"  ⚠️  SQLite chunk storage failed: {e}")
            
            # Step 5: Index chunks in VectorStore (Qdrant)
            if vector_store:
                try:
                    vector_store.add_chunks(
                        chunks,
                        domain=domain,
                        doc_metadata={
                            'title': doc.title or 'Unknown',
                            'source_path': str(pdf_path)
                        }
                    )
                    logger.info(f"  🧲 Embedded {len(chunks)} chunks in Qdrant")
                except Exception as e:
                    logger.warning(f"  ⚠️  Qdrant indexing failed: {e}")
            
            # Step 6: Extract entities (requires LLM) - PAPER-BASED
            extraction_result = None
            if entity_extractor:
                try:
                    # Use paper-based extraction (recommended)
                    extraction_result = entity_extractor.extract_from_paper(
                        chunks,
                        paper_metadata={
                            'title': doc.title,
                            'authors': doc.authors or [],
                            'year': doc.year
                        },
                        context_window=2048,  # Adjust based on your model
                        max_output_tokens=500
                    )
                    
                    total_entities = (
                        len(extraction_result.species) + 
                        len(extraction_result.measurements) + 
                        len(extraction_result.locations)
                    )
                    logger.info(f"  🧬 Extracted {total_entities} entities from paper")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  Entity extraction failed: {e}")
            
            # Step 6b: Extract citations (NEW)
            citations = []
            if citation_extractor:
                try:
                    citations = citation_extractor.extract_citations(chunks, doc.doc_id)
                    if citations:
                        logger.info(f"  📚 Extracted {len(citations)} citations")
                except Exception as e:
                    logger.warning(f"  ⚠️  Citation extraction failed: {e}")
            
            # Step 7: Build knowledge graph (Neo4j)
            if graph_builder:
                try:
                    # Build complete metadata dict for Neo4j
                    paper_meta = {
                        'title': indexed.title or doc.title or 'Untitled',
                        'authors': doc.authors or [],
                        'year': indexed.year or getattr(doc, 'year', None),
                        'doi': getattr(doc, 'doi', None),
                        'abstract': doc.abstract,
                        'source_path': str(doc.source_path) if doc.source_path else None,
                    }
                    
                    # Always create paper node with full metadata
                    graph_builder.add_paper(
                        doc_id=doc.doc_id,
                        title=paper_meta['title'],
                        authors=paper_meta['authors'],
                        year=paper_meta['year'],
                        doi=paper_meta['doi'],
                        abstract=paper_meta['abstract'],
                        source_path=paper_meta['source_path'],
                    )
                    
                    # Add extraction results (entities/relations) WITH metadata
                    if extraction_result:
                        graph_builder.add_extraction_result(
                            doc_id=doc.doc_id,
                            result=extraction_result,
                            paper_metadata=paper_meta,
                        )
                    
                    # Add citations
                    if citations:
                        graph_builder.add_citations(doc.doc_id, citations)
                    
                    logger.info(f"  🕸️  Added to knowledge graph")
                except Exception as e:
                    logger.warning(f"  ⚠️  Graph building failed: {e}")
            
            elapsed = time.time() - t0
            logger.info(f"  ⏱️  Completed in {elapsed:.1f}s")
            processed += 1
            
        except Exception as e:
            logger.error(f"  ❌ FAILED: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 INGESTION COMPLETE")
    logger.info(f"  Processed: {processed}/{len(pdf_paths)} PDFs")
    logger.info(f"  Failed:    {failed}")
    logger.info(f"  Chunks:    {total_chunks}")
    logger.info(f"  SQLite:    {paper_index.count()} papers indexed")
    
    if vector_store:
        try:
            stats = vector_store.get_stats()
            logger.info(f"  Qdrant:    {stats.get('vectors_count', '?')} vectors")
        except Exception:
            pass
    
    if graph_builder:
        try:
            stats = graph_builder.get_stats()
            logger.info(f"  Neo4j:     {stats.paper_count} papers, {stats.species_count} species")
        except Exception:
            pass
    
    # Cleanup
    if vector_store:
        try:
            vector_store.close()
        except Exception:
            pass
    if graph_builder:
        try:
            graph_builder.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="EcoloGRAPH Ingestion Pipeline - Process PDFs into searchable knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py data/raw/               # All PDFs
  python scripts/ingest.py paper.pdf                # Single PDF
  python scripts/ingest.py data/raw/ --skip-graph   # No Neo4j
  python scripts/ingest.py data/raw/ --skip-extract # No LLM entity extraction
        """
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip entity extraction (no LLM required)"
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip Neo4j graph building"
    )
    parser.add_argument(
        "--skip-vectors", action="store_true",
        help="Skip Qdrant vector indexing"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Overlap between chunks (default: 200)"
    )
    parser.add_argument(
        "--restart-interval",
        type=int,
        default=20,
        help="Restart LM Studio every N papers to prevent memory leaks (0 = disabled, default: 20)"
    )
    
    # LLM configuration
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama model for ingestion (default: from config, e.g. qwen3:8b)"
    )
    parser.add_argument(
        "--thinking", action="store_true", default=False,
        help="Enable thinking mode (slower, better reasoning for ambiguous entities)"
    )
    parser.add_argument(
        "--no-thinking", action="store_true", default=False,
        help="Explicitly disable thinking mode (fast JSON extraction, default)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max tokens per LLM response (default: 2048)"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="LLM request timeout in seconds (default: 120)"
    )
    
    args = parser.parse_args()
    
    # Resolve thinking flag
    thinking = args.thinking and not args.no_thinking
    
    ingest(
        input_path=args.input_path,
        skip_extract=args.skip_extract,
        skip_graph=args.skip_graph,
        skip_vectors=args.skip_vectors,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        restart_interval=args.restart_interval,
        model=args.model,
        thinking=thinking,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
