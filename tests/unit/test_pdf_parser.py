"""
Unit tests for PDF parser module.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.pdf_parser import PDFParser, ParsedDocument, TableData, ParsedSection


class TestPDFParser:
    """Tests for PDFParser class."""
    
    def test_parser_initialization(self):
        """Test that parser initializes correctly."""
        parser = PDFParser()
        assert parser.converter is not None
    
    def test_parse_nonexistent_file(self):
        """Test that parsing a nonexistent file raises error."""
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(Path("/nonexistent/file.pdf"))
    
    def test_parse_non_pdf_file(self, tmp_path):
        """Test that parsing a non-PDF file raises error."""
        # Create a temp text file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is not a PDF")
        
        parser = PDFParser()
        with pytest.raises(ValueError, match="Not a PDF file"):
            parser.parse(txt_file)
    
    def test_generate_doc_id_consistency(self, tmp_path):
        """Test that same file produces same doc_id."""
        # Create a temp PDF-like file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 test content")
        
        parser = PDFParser()
        id1 = parser._generate_doc_id(pdf_file)
        id2 = parser._generate_doc_id(pdf_file)
        
        assert id1 == id2
        assert len(id1) == 12  # MD5 hash truncated to 12 chars
    
    def test_generate_doc_id_different_files(self, tmp_path):
        """Test that different files produce different doc_ids."""
        pdf1 = tmp_path / "test1.pdf"
        pdf1.write_bytes(b"%PDF-1.4 content one")
        
        pdf2 = tmp_path / "test2.pdf"
        pdf2.write_bytes(b"%PDF-1.4 content two")
        
        parser = PDFParser()
        id1 = parser._generate_doc_id(pdf1)
        id2 = parser._generate_doc_id(pdf2)
        
        assert id1 != id2


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""
    
    def test_parsed_document_creation(self):
        """Test creating a ParsedDocument."""
        doc = ParsedDocument(
            doc_id="test123",
            source_path=Path("/test/path.pdf"),
            title="Test Paper",
            authors=["John Doe", "Jane Smith"],
            num_pages=10
        )
        
        assert doc.doc_id == "test123"
        assert doc.title == "Test Paper"
        assert len(doc.authors) == 2
        assert doc.num_pages == 10
        assert doc.sections == []
        assert doc.tables == []
    
    def test_parsed_document_defaults(self):
        """Test ParsedDocument default values."""
        doc = ParsedDocument(
            doc_id="test",
            source_path=Path("/test.pdf")
        )
        
        assert doc.title is None
        assert doc.authors == []
        assert doc.abstract is None
        assert doc.doi is None
        assert doc.full_text == ""
        assert doc.num_pages == 0


class TestTableData:
    """Tests for TableData dataclass."""
    
    def test_table_data_creation(self):
        """Test creating TableData."""
        table = TableData(
            table_id="table_1",
            page=5,
            caption="Sample sizes by location",
            dataframe=None,
            markdown="| Location | N |\n|---|---|\n| A | 10 |"
        )
        
        assert table.table_id == "table_1"
        assert table.page == 5
        assert table.caption == "Sample sizes by location"
        assert "Location" in table.markdown


class TestParsedSection:
    """Tests for ParsedSection dataclass."""
    
    def test_section_creation(self):
        """Test creating ParsedSection."""
        section = ParsedSection(
            title="Methods",
            level=2,
            text="We collected samples from...",
            page_start=3,
            page_end=5
        )
        
        assert section.title == "Methods"
        assert section.level == 2
        assert "samples" in section.text
