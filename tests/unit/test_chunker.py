"""
Unit tests for document chunker module.
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.chunker import DocumentChunker, DocumentChunk
from src.ingestion.pdf_parser import ParsedDocument, ParsedSection


@pytest.fixture
def sample_document():
    """Create a sample parsed document for testing."""
    return ParsedDocument(
        doc_id="test_doc_123",
        source_path=Path("/test/sample_paper.pdf"),
        title="Depth Distribution of Atlantic Cod",
        authors=["Smith, J.", "García, M."],
        doi="10.1234/test.2023",
        full_text="""# Abstract

This study examines the depth distribution of Atlantic cod (Gadus morhua) 
in the North Sea. We found significant seasonal variation in depth preferences.

# Introduction

The Atlantic cod is one of the most commercially important fish species 
in the North Atlantic. Understanding their depth distribution is crucial 
for fisheries management and conservation efforts.

Previous studies have shown that cod exhibit complex vertical migration patterns.
These patterns are influenced by temperature, prey availability, and reproductive needs.

# Methods

We used acoustic telemetry to track 150 individual cod over a 2-year period.
Fish were tagged at multiple locations across the North Sea.
Depth data was recorded every 30 minutes.

## Study Area

The study was conducted in the central North Sea, between 54°N and 58°N latitude.

## Data Analysis

We used mixed-effects models to analyze depth preferences.
Temperature and season were included as fixed effects.

# Results

Adult cod were found at depths ranging from 50-200m (Table 1).
Juvenile cod preferred shallower waters, typically 20-80m.

The deepest recordings exceeded 300m during winter months.

# Discussion

Our findings confirm the importance of depth as a habitat variable for cod.
""",
        sections=[
            ParsedSection(title="Abstract", level=1, text="This study examines...", page_start=1, page_end=1),
            ParsedSection(title="Introduction", level=1, text="The Atlantic cod is...", page_start=1, page_end=2),
            ParsedSection(title="Methods", level=1, text="We used acoustic telemetry...", page_start=2, page_end=3),
            ParsedSection(title="Results", level=1, text="Adult cod were found...", page_start=3, page_end=4),
            ParsedSection(title="Discussion", level=1, text="Our findings confirm...", page_start=4, page_end=5),
        ],
        num_pages=5
    )


class TestDocumentChunker:
    """Tests for DocumentChunker class."""
    
    def test_chunker_initialization(self):
        """Test that chunker initializes with correct defaults."""
        chunker = DocumentChunker()
        
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.min_chunk_size == 100
        assert chunker.respect_sections is True
    
    def test_chunker_custom_params(self):
        """Test chunker with custom parameters."""
        chunker = DocumentChunker(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            respect_sections=False
        )
        
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.min_chunk_size == 50
        assert chunker.respect_sections is False
    
    def test_chunk_document_produces_chunks(self, sample_document):
        """Test that chunking produces non-empty list of chunks."""
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(sample_document)
        
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_chunks_have_metadata(self, sample_document):
        """Test that chunks preserve document metadata."""
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(sample_document)
        
        for chunk in chunks:
            assert chunk.doc_id == sample_document.doc_id
            assert chunk.doc_title == sample_document.title
            assert chunk.doc_doi == sample_document.doi
    
    def test_chunks_have_unique_ids(self, sample_document):
        """Test that each chunk has a unique ID."""
        chunker = DocumentChunker(chunk_size=500)
        chunks = chunker.chunk_document(sample_document)
        
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique
    
    def test_chunks_have_word_counts(self, sample_document):
        """Test that chunks have accurate word counts."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(sample_document)
        
        for chunk in chunks:
            assert chunk.word_count > 0
            assert chunk.char_count > 0
            # Word count should match actual words
            assert chunk.word_count == len(chunk.text.split())


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            chunk_id="chunk_abc123",
            doc_id="doc_xyz",
            text="This is sample text about fish depth.",
            section="Results",
            section_level=1,
            chunk_idx=0,
            char_count=38,
            word_count=7
        )
        
        assert chunk.chunk_id == "chunk_abc123"
        assert chunk.doc_id == "doc_xyz"
        assert "fish" in chunk.text
        assert chunk.section == "Results"
    
    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = DocumentChunk(
            chunk_id="test123",
            doc_id="doc456",
            text="Test text",
            section="Methods",
            chunk_idx=0
        )
        
        d = chunk.to_dict()
        
        assert isinstance(d, dict)
        assert d["chunk_id"] == "test123"
        assert d["doc_id"] == "doc456"
        assert d["section"] == "Methods"
    
    def test_chunk_detects_table_reference(self):
        """Test that chunk detects table references in text."""
        from src.ingestion.chunker import DocumentChunker
        from src.ingestion.pdf_parser import ParsedDocument
        
        doc = ParsedDocument(
            doc_id="test",
            source_path=Path("/test.pdf"),
            full_text="Results are shown in Table 1. The data indicates significant differences between groups. " * 3
        )
        
        chunker = DocumentChunker(chunk_size=2000)
        chunks = chunker.chunk_document(doc)
        
        # At least one chunk should have has_table=True
        assert any(c.has_table for c in chunks)
    
    def test_chunk_detects_figure_reference(self):
        """Test that chunk detects figure references in text."""
        from src.ingestion.chunker import DocumentChunker
        from src.ingestion.pdf_parser import ParsedDocument
        
        doc = ParsedDocument(
            doc_id="test",
            source_path=Path("/test.pdf"),
            full_text="As shown in Figure 2, the distribution varies significantly across different regions and seasons. " * 3
        )
        
        chunker = DocumentChunker(chunk_size=2000)
        chunks = chunker.chunk_document(doc)
        
        # At least one chunk should have has_figure=True
        assert any(c.has_figure for c in chunks)
