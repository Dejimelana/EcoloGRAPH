"""
Document chunking module for EcoloGRAPH.

Splits parsed documents into chunks suitable for embedding and retrieval,
while preserving context and metadata for traceability.
"""
import logging
from dataclasses import dataclass, field
from typing import Iterator
import hashlib
import re

from .pdf_parser import ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of document text with full metadata for traceability.
    
    Each chunk knows exactly where it came from in the original document.
    """
    # Identification
    chunk_id: str
    doc_id: str
    
    # Content
    text: str
    
    # Location in document
    page: int | None = None
    section: str | None = None
    section_level: int | None = None
    paragraph_idx: int | None = None
    chunk_idx: int = 0  # Index within the document
    
    # Document metadata (copied for convenience)
    doc_title: str | None = None
    doc_authors: list[str] = field(default_factory=list)
    doc_year: int | None = None
    doc_doi: str | None = None
    source_path: str | None = None
    
    # Chunk metadata
    char_count: int = 0
    word_count: int = 0
    has_table: bool = False
    has_figure: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "section_level": self.section_level,
            "paragraph_idx": self.paragraph_idx,
            "chunk_idx": self.chunk_idx,
            "doc_title": self.doc_title,
            "doc_authors": self.doc_authors,
            "doc_year": self.doc_year,
            "doc_doi": self.doc_doi,
            "source_path": self.source_path,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "has_table": self.has_table,
            "has_figure": self.has_figure,
        }


class DocumentChunker:
    """
    Splits documents into chunks for embedding.
    
    Strategies:
    - Section-based: Keep sections together (preferred for structure)
    - Size-based: Split by character count with overlap
    - Hybrid: Section-aware size-based splitting
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_sections: bool = True
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
            respect_sections: If True, prefer to break at section boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sections = respect_sections
        
        logger.info(
            f"DocumentChunker initialized: size={chunk_size}, "
            f"overlap={chunk_overlap}, respect_sections={respect_sections}"
        )
    
    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Split a parsed document into chunks.
        
        Args:
            doc: Parsed document to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_idx = 0
        
        if self.respect_sections and doc.sections:
            # Section-aware chunking
            for section in doc.sections:
                section_chunks = self._chunk_section(
                    text=section.text,
                    section_title=section.title,
                    section_level=section.level,
                    doc=doc,
                    start_chunk_idx=chunk_idx
                )
                chunks.extend(section_chunks)
                chunk_idx += len(section_chunks)
        else:
            # Simple text chunking
            chunks = self._chunk_text(
                text=doc.full_text,
                doc=doc,
                start_chunk_idx=0
            )
        
        # Add table chunks
        for table in doc.tables:
            chunk = self._create_table_chunk(table, doc, len(chunks))
            if chunk:
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {doc.source_path.name}")
        
        return chunks
    
    def chunk_with_hierarchy(
        self, 
        doc: ParsedDocument
    ) -> tuple[HierarchicalChunk, list[HierarchicalChunk]]:
        """
        Create hierarchical chunks: one parent (full document) + many children (sections).
        
        This strategy enables:
        - Entity extraction from FULL document context (parent)
        - Semantic search on granular chunks (children)
        - No loss of cross-section relationships
        
        Args:
            doc: Parsed document to chunk
            
        Returns:
            Tuple of (parent_chunk, child_chunks)
            - parent_chunk: Full document text for entity extraction
            - child_chunks: Section-based chunks for embeddings
        """
        # Build parent text (full document with structure)
        parent_text = self._build_parent_text(doc)
        
        # Create parent chunk
        parent_chunk_id = hashlib.md5(
            f"{doc.doc_id}:parent".encode()
        ).hexdigest()[:16]
        
        parent_chunk = HierarchicalChunk(
            chunk_id=parent_chunk_id,
            doc_id=doc.doc_id,
            text=parent_text,
            doc_title=doc.title,
            doc_authors=doc.authors,
            doc_year=doc.year,
            doc_doi=doc.doi,
            source_path=str(doc.source_path),
            char_count=len(parent_text),
            word_count=len(parent_text.split()),
            parent_id=None,  # Parent has no parent
            is_parent=True
        )
        
        # Create child chunks (using existing section-aware logic)
        child_chunks = []
        chunk_idx = 0
        
        if self.respect_sections and doc.sections:
            for section in doc.sections:
                section_chunks = self._chunk_section(
                    text=section.text,
                    section_title=section.title,
                    section_level=section.level,
                    doc=doc,
                    start_chunk_idx=chunk_idx
                )
                # Convert to HierarchicalChunk with parent link
                for chunk in section_chunks:
                    hierarchical_chunk = HierarchicalChunk(
                        **chunk.__dict__,
                        parent_id=parent_chunk_id,
                        is_parent=False
                    )
                    child_chunks.append(hierarchical_chunk)
                chunk_idx += len(section_chunks)
        else:
            # Fallback: simple text chunking
            simple_chunks = self._chunk_text(
                text=doc.full_text,
                doc=doc,
                start_chunk_idx=0
            )
            child_chunks = [
                HierarchicalChunk(
                    **chunk.__dict__,
                    parent_id=parent_chunk_id,
                    is_parent=False
                )
                for chunk in simple_chunks
            ]
        
        logger.info(
            f"Hierarchical chunking: 1 parent + {len(child_chunks)} children "
            f"from {doc.source_path.name}"
        )
        
        return parent_chunk, child_chunks
    
    def _build_parent_text(self, doc: ParsedDocument) -> str:
        """
        Build structured full-text representation of document for entity extraction.
        Preserves section structure and table context.
        """
        parts = []
        
        # Add metadata header
        parts.append(f"TITLE: {doc.title}")
        if doc.authors:
            parts.append(f"AUTHORS: {', '.join(doc.authors)}")
        if doc.year:
            parts.append(f"YEAR: {doc.year}")
        parts.append("\n" + "="*80 + "\n")
        
        # Add abstract if available
        if doc.abstract:
            parts.append("ABSTRACT\n")
            parts.append(doc.abstract)
            parts.append("\n" + "-"*80 + "\n")
        
        # Add sections with clear markers
        if doc.sections:
            for section in doc.sections:
                header = "#" * section.level + " " + section.title
                parts.append(f"\n{header}\n")
                parts.append(section.text)
        else:
            # Fallback: use full_text
            parts.append(doc.full_text)
        
        # Add tables with context
        if doc.tables:
            parts.append("\n" + "="*80 + "\nTABLES\n")
            for table in doc.tables:
                if table.markdown:
                    parts.append(f"\nTable {table.table_id}")
                    if table.caption:
                        parts.append(f": {table.caption}")
                    parts.append(f"\n{table.markdown}\n")
        
        return "\n".join(parts)
    
@dataclass
class HierarchicalChunk(DocumentChunk):
    """
    Extended chunk with parent-child hierarchy for improved entity extraction.
    
    Child chunks preserve granular traceability while parent_id links to 
    full document context for LLM extraction.
    """
    parent_id: str | None = None  # Link to full document parent
    is_parent: bool = False  # True for full-document parent chunks


    def _chunk_section(
        self,
        text: str,
        section_title: str,
        section_level: int,
        doc: ParsedDocument,
        start_chunk_idx: int
    ) -> list[DocumentChunk]:
        """Chunk a single section, respecting size limits."""
        if not text.strip():
            return []
        
        # If section fits in one chunk, keep it together
        if len(text) <= self.chunk_size:
            return [self._create_chunk(
                text=text,
                doc=doc,
                section=section_title,
                section_level=section_level,
                chunk_idx=start_chunk_idx
            )]
        
        # Otherwise, split by paragraphs first, then by size
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_text = ""
        current_para_idx = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds limit
            if len(current_text) + len(para) > self.chunk_size and current_text:
                # Save current chunk
                chunks.append(self._create_chunk(
                    text=current_text.strip(),
                    doc=doc,
                    section=section_title,
                    section_level=section_level,
                    paragraph_idx=current_para_idx,
                    chunk_idx=start_chunk_idx + len(chunks)
                ))
                # Start new chunk with overlap
                overlap_text = current_text[-self.chunk_overlap:] if len(current_text) > self.chunk_overlap else ""
                current_text = overlap_text + para + "\n\n"
            else:
                current_text += para + "\n\n"
            
            current_para_idx += 1
        
        # Don't forget the last chunk
        if current_text.strip() and len(current_text.strip()) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                text=current_text.strip(),
                doc=doc,
                section=section_title,
                section_level=section_level,
                paragraph_idx=current_para_idx,
                chunk_idx=start_chunk_idx + len(chunks)
            ))
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        doc: ParsedDocument,
        start_chunk_idx: int
    ) -> list[DocumentChunk]:
        """Simple text chunking without section awareness."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                sentence_end = self._find_sentence_boundary(text, end - 100, end + 100)
                if sentence_end > start:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    doc=doc,
                    chunk_idx=start_chunk_idx + len(chunks)
                ))
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        doc: ParsedDocument,
        chunk_idx: int,
        section: str | None = None,
        section_level: int | None = None,
        paragraph_idx: int | None = None,
        page: int | None = None
    ) -> DocumentChunk:
        """Create a DocumentChunk with all metadata."""
        # Generate chunk ID
        chunk_id = hashlib.md5(
            f"{doc.doc_id}:{chunk_idx}:{text[:50]}".encode()
        ).hexdigest()[:16]
        
        # Check for table/figure references in text
        has_table = bool(re.search(r'table\s*\d+', text.lower()))
        has_figure = bool(re.search(r'fig(ure)?\s*\.?\s*\d+', text.lower()))
        
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            text=text,
            page=page,
            section=section,
            section_level=section_level,
            paragraph_idx=paragraph_idx,
            chunk_idx=chunk_idx,
            doc_title=doc.title,
            doc_authors=doc.authors,
            doc_doi=doc.doi,
            source_path=str(doc.source_path),
            char_count=len(text),
            word_count=len(text.split()),
            has_table=has_table,
            has_figure=has_figure
        )
    
    def _create_table_chunk(
        self,
        table,
        doc: ParsedDocument,
        chunk_idx: int
    ) -> DocumentChunk | None:
        """Create a chunk from a table."""
        if not table.markdown:
            return None
        
        text = f"Table {table.table_id}"
        if table.caption:
            text += f": {table.caption}"
        text += f"\n\n{table.markdown}"
        
        chunk = self._create_chunk(
            text=text,
            doc=doc,
            chunk_idx=chunk_idx,
            page=table.page
        )
        chunk.has_table = True
        
        return chunk
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find a sentence boundary within the given range."""
        search_text = text[max(0, start):min(len(text), end)]
        
        # Look for sentence-ending punctuation followed by space
        patterns = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
        
        best_pos = -1
        for pattern in patterns:
            pos = search_text.rfind(pattern)
            if pos > best_pos:
                best_pos = pos
        
        if best_pos > 0:
            return start + best_pos + 2  # Include the punctuation and space
        
        return end  # Fall back to original end
