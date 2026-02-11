"""
PDF Parser module using Docling for scientific papers.

Extracts text, tables, and metadata from PDF documents with
position information for citation traceability.
"""
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
import hashlib

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.document import ConversionResult

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Extracted table with metadata."""
    table_id: str
    page: int
    caption: str | None
    dataframe: "pd.DataFrame | None"  # Will be pandas DataFrame
    markdown: str
    

@dataclass
class FigureData:
    """Extracted figure reference with metadata."""
    figure_id: str
    page: int
    caption: str | None


@dataclass
class ParsedSection:
    """A section of the document with its content."""
    title: str
    level: int  # 1 = h1, 2 = h2, etc.
    text: str
    page_start: int
    page_end: int


@dataclass
class ParsedDocument:
    """Complete parsed document with all extracted content."""
    # Identification
    doc_id: str
    source_path: Path
    
    # Basic metadata (extracted from PDF)
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    year: int | None = None  # Publication year
    
    # Content
    full_text: str = ""
    sections: list[ParsedSection] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    figures: list[FigureData] = field(default_factory=list)
    
    # Metadata
    num_pages: int = 0
    
    def get_text_by_page(self, page: int) -> str:
        """Get text content for a specific page."""
        # This would require page-level tracking during parsing
        # For now, return empty - will be enhanced
        return ""


class PDFParser:
    """
    PDF parser using Docling for scientific papers.
    
    Extracts:
    - Full text with section structure
    - Tables as structured data
    - Figure references
    - Basic metadata (title, authors if detectable)
    """
    
    def __init__(self):
        """Initialize the PDF parser with Docling converter."""
        self.converter = DocumentConverter()
        logger.info("PDFParser initialized with Docling")
    
    def parse(self, pdf_path: Path | str) -> ParsedDocument:
        """
        Parse a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument with all extracted content
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        logger.info(f"Parsing PDF: {pdf_path.name}")
        
        # Generate document ID from file hash
        doc_id = self._generate_doc_id(pdf_path)
        
        # Convert with Docling
        try:
            result: ConversionResult = self.converter.convert(str(pdf_path))
        except Exception as e:
            logger.error(f"Docling conversion failed for {pdf_path}: {e}")
            raise
        
        # Extract content
        document = result.document
        
        # Build parsed document
        parsed = ParsedDocument(
            doc_id=doc_id,
            source_path=pdf_path,
            num_pages=getattr(document, 'num_pages', 0) or self._count_pages(result)
        )
        
        # Extract text
        parsed.full_text = document.export_to_markdown()
        
        # Extract metadata
        self._extract_metadata(document, parsed)
        
        # Extract sections
        parsed.sections = self._extract_sections(document)
        
        # Extract tables
        parsed.tables = self._extract_tables(document)
        
        # Extract figures
        parsed.figures = self._extract_figures(document)
        
        logger.info(
            f"Parsed {pdf_path.name}: {len(parsed.sections)} sections, "
            f"{len(parsed.tables)} tables, {len(parsed.figures)} figures"
        )
        
        return parsed
    
    def parse_batch(
        self, 
        pdf_paths: list[Path | str]
    ) -> Generator[ParsedDocument, None, None]:
        """
        Parse multiple PDFs, yielding results as they complete.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Yields:
            ParsedDocument for each successfully parsed file
        """
        for pdf_path in pdf_paths:
            try:
                yield self.parse(pdf_path)
            except Exception as e:
                logger.error(f"Failed to parse {pdf_path}: {e}")
                continue
    
    def _generate_doc_id(self, pdf_path: Path) -> str:
        """Generate a unique document ID from file content hash."""
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    
    def _count_pages(self, result: ConversionResult) -> int:
        """Count pages from conversion result."""
        try:
            # Try to get page count from document metadata
            if hasattr(result.document, 'pages'):
                return len(result.document.pages)
        except:
            pass
        return 0
    
    def _extract_metadata(self, document, parsed: ParsedDocument) -> None:
        """Extract title, authors, abstract from document."""
        import re
        
        # --- Title extraction (multi-level fallback) ---
        
        # Level 1: Docling metadata
        if hasattr(document, 'title') and document.title:
            parsed.title = document.title
        
        # Level 2: First markdown heading (# Title)
        if not parsed.title and parsed.full_text:
            # Look for first # heading in the markdown export
            heading_match = re.search(r'^#\s+(.+)$', parsed.full_text, re.MULTILINE)
            if heading_match:
                title_candidate = heading_match.group(1).strip()
                # Headings like "Abstract", "Introduction" are not titles
                skip_headings = {
                    'abstract', 'introduction', 'methods', 'results',
                    'discussion', 'references', 'acknowledgments',
                    'supplementary', 'table of contents', 'contents',
                }
                if title_candidate.lower() not in skip_headings and len(title_candidate) > 5:
                    parsed.title = title_candidate
        
        # Level 3: First non-empty line of text (often the title in papers)
        if not parsed.title and parsed.full_text:
            lines = parsed.full_text.strip().split('\n')
            for line in lines[:10]:  # Check first 10 lines
                cleaned = line.strip().lstrip('#').strip()
                # Skip very short lines, markdown formatting, empty lines
                if (cleaned 
                    and len(cleaned) > 10 
                    and len(cleaned) < 300
                    and not cleaned.startswith('!')    # images
                    and not cleaned.startswith('|')    # tables
                    and not cleaned.startswith('```')  # code
                    and not cleaned.startswith('---')  # dividers
                    and cleaned.lower() not in {'abstract', 'introduction'}
                ):
                    # Remove markdown bold/italic
                    cleaned = re.sub(r'[*_]{1,3}', '', cleaned).strip()
                    if cleaned:
                        parsed.title = cleaned
                        break
        
        # Level 4: Filename fallback (clean up the filename into a readable title)
        if not parsed.title and parsed.source_path:
            stem = Path(parsed.source_path).stem
            # Remove year and journal suffix (ScienceDirect pattern)
            year_match = re.search(r'_(\d{4})_', stem)
            if year_match:
                stem = stem[:year_match.start()]
            stem = re.sub(r'---+', ' — ', stem)
            stem = re.sub(r'--', ' – ', stem)
            stem = stem.replace('-', ' ').replace('_', ' ')
            stem = re.sub(r'\s+', ' ', stem).strip()
            if stem:
                parsed.title = stem[0].upper() + stem[1:]
        
        # --- Authors ---
        if hasattr(document, 'authors') and document.authors:
            parsed.authors = list(document.authors)
        
        # --- Abstract extraction ---
        if parsed.full_text:
            abstract = None
            text = parsed.full_text
            
            # Strategy 1: Look for a markdown heading "## Abstract" or "# Abstract"
            abs_heading = re.search(
                r'^#{1,3}\s+[Aa]bstract\s*\n(.*?)(?=\n#{1,3}\s|\Z)',
                text, re.DOTALL | re.MULTILINE
            )
            if abs_heading:
                abstract = abs_heading.group(1).strip()
            
            # Strategy 2: Look for "Abstract" as a standalone label in the first 30% of text
            # FIXED: Added References, Bibliography, Acknowledgments as stop markers
            if not abstract:
                first_third = text[:len(text) // 3]
                # Match "Abstract" at start of line followed by paragraph text
                abs_label = re.search(
                    r'(?:^|\n)\s*[Aa]bstract[:\.\s]*\n(.*?)(?=\n\s*(?:Keywords?|Introduction|[Rr]eferences?|[Bb]ibliography|[Aa]cknowledg|1[\.\s]|#{1,3}\s)|\Z)',
                    first_third, re.DOTALL
                )
                if abs_label:
                    abstract = abs_label.group(1).strip()
            
            # Strategy 3: "Abstract" followed by text on the same or next line (first 30%)
            # FIXED: Added References, Bibliography as stop markers
            if not abstract:
                abs_inline = re.search(
                    r'[Aa]bstract[:\.\s—–-]*\s*(.{50,}?)(?=\n\s*(?:Keywords?|Introduction|[Rr]eferences?|[Bb]ibliography|[Aa]cknowledg|1[\.\s])|\n\n\n)',
                    first_third if 'first_third' in dir() else text[:len(text) // 3],
                    re.DOTALL
                )
                if abs_inline:
                    abstract = abs_inline.group(1).strip()
            
            if abstract:
                # Clean up: remove excessive whitespace
                abstract = re.sub(r'\s+', ' ', abstract).strip()
                
                # IMPROVED: Better length validation
                # Typical abstract: 150-300 words (~800-1600 chars)
                # Reject if too short (likely just header) or way too long (captured references)
                word_count = len(abstract.split())
                
                if word_count < 20:
                    # Too short, likely not a real abstract
                    logger.warning(f"Abstract too short ({word_count} words), discarding")
                    abstract = None
                elif word_count > 800:
                    # Way too long - likely captured references
                    # Check if "References" or common ref patterns appear
                    if any(pattern in abstract for pattern in [
                        'References', 'REFERENCES', 'Bibliography', 'BIBLIOGRAPHY',
                        '- ', '• ', 'et al.,', '. doi:', 'http://dx.doi',
                        'Retrieved from', 'Downloaded from'
                    ]):
                        logger.warning(f"Abstract appears to contain references ({word_count} words), truncating")
                        # Find first occurrence of reference markers
                        ref_start = min([
                            abstract.find(pattern) for pattern in [
                                'References', 'REFERENCES', 'Bibliography',
                                'Acknowledgment'
                            ] if pattern in abstract
                        ] + [len(abstract)])
                        
                        abstract = abstract[:ref_start].strip()
                        word_count = len(abstract.split())
                    
                    # Final truncation if still too long
                    if word_count > 500:
                        # Truncate at ~400 words (reasonable max for an abstract)
                        words = abstract.split()[:400]
                        abstract = ' '.join(words) + "..."
                
                # Only save if reasonable length
                if abstract and len(abstract.split()) >= 20:
                    parsed.abstract = abstract
                else:
                    parsed.abstract = None
        
        # --- Year extraction from text ---
        if not hasattr(parsed, '_year_extracted') and parsed.full_text:
            # Try to find year in the first ~500 chars (often in header/citation)
            header_text = parsed.full_text[:500]
            year_matches = re.findall(r'\b(20[0-2]\d|19[89]\d)\b', header_text)
            if year_matches:
                # Store as attribute for the ingestion script to use
                parsed._extracted_year = int(year_matches[0])
    
    def _extract_sections(self, document) -> list[ParsedSection]:
        """Extract document sections with hierarchy."""
        sections = []
        
        # Parse markdown for headers
        lines = document.export_to_markdown().split('\n')
        current_section = None
        current_text = []
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    current_section.text = '\n'.join(current_text).strip()
                    sections.append(current_section)
                
                # Count header level
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = ParsedSection(
                    title=title,
                    level=level,
                    text="",
                    page_start=0,  # Would need page tracking
                    page_end=0
                )
                current_text = []
            else:
                current_text.append(line)
        
        # Don't forget last section
        if current_section:
            current_section.text = '\n'.join(current_text).strip()
            sections.append(current_section)
        
        return sections
    
    def _extract_tables(self, document) -> list[TableData]:
        """Extract tables from document."""
        tables = []
        
        if hasattr(document, 'tables'):
            for idx, table in enumerate(document.tables):
                try:
                    table_data = TableData(
                        table_id=f"table_{idx + 1}",
                        page=getattr(table, 'page', 0),
                        caption=getattr(table, 'caption', None),
                        dataframe=table.export_to_dataframe() if hasattr(table, 'export_to_dataframe') else None,
                        markdown=table.export_to_markdown() if hasattr(table, 'export_to_markdown') else str(table)
                    )
                    tables.append(table_data)
                except Exception as e:
                    logger.warning(f"Failed to extract table {idx}: {e}")
        
        return tables
    
    def _extract_figures(self, document) -> list[FigureData]:
        """Extract figure references from document."""
        figures = []
        
        if hasattr(document, 'pictures') or hasattr(document, 'figures'):
            fig_list = getattr(document, 'pictures', None) or getattr(document, 'figures', [])
            for idx, fig in enumerate(fig_list):
                try:
                    fig_data = FigureData(
                        figure_id=f"figure_{idx + 1}",
                        page=getattr(fig, 'page', 0),
                        caption=getattr(fig, 'caption', None)
                    )
                    figures.append(fig_data)
                except Exception as e:
                    logger.warning(f"Failed to extract figure {idx}: {e}")
        
        return figures
