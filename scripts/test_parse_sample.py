"""
Script to test PDF parsing with a real PDF file.

Usage:
    python scripts/test_parse_sample.py --input path/to/paper.pdf
    python scripts/test_parse_sample.py --input data/raw  # Process all PDFs in directory
"""
import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import PDFParser, DocumentChunker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_single_pdf(pdf_path: Path, chunker: DocumentChunker):
    """Test parsing a single PDF file."""
    print(f"\n{'='*60}")
    print(f"📄 Parsing: {pdf_path.name}")
    print('='*60)
    
    parser = PDFParser()
    
    try:
        # Parse the PDF
        doc = parser.parse(pdf_path)
        
        print(f"\n✅ Successfully parsed!")
        print(f"   Doc ID: {doc.doc_id}")
        print(f"   Title: {doc.title or '(not detected)'}")
        print(f"   Authors: {', '.join(doc.authors) if doc.authors else '(not detected)'}")
        print(f"   Pages: {doc.num_pages}")
        print(f"   Sections: {len(doc.sections)}")
        print(f"   Tables: {len(doc.tables)}")
        print(f"   Figures: {len(doc.figures)}")
        
        # Show sections
        if doc.sections:
            print(f"\n   📑 Sections:")
            for section in doc.sections[:5]:  # First 5
                print(f"      - {section.title} (level {section.level})")
            if len(doc.sections) > 5:
                print(f"      ... and {len(doc.sections) - 5} more")
        
        # Show tables
        if doc.tables:
            print(f"\n   📊 Tables:")
            for table in doc.tables[:3]:  # First 3
                caption = table.caption[:50] + "..." if table.caption and len(table.caption) > 50 else table.caption
                print(f"      - {table.table_id}: {caption or '(no caption)'}")
        
        # Show abstract preview
        if doc.abstract:
            abstract_preview = doc.abstract[:200] + "..." if len(doc.abstract) > 200 else doc.abstract
            print(f"\n   📝 Abstract preview:")
            print(f"      {abstract_preview}")
        
        # Chunk the document
        chunks = chunker.chunk_document(doc)
        print(f"\n   ✂️  Created {len(chunks)} chunks")
        
        # Show chunk statistics
        if chunks:
            avg_words = sum(c.word_count for c in chunks) / len(chunks)
            print(f"      Average words per chunk: {avg_words:.0f}")
            print(f"      Chunks with table refs: {sum(1 for c in chunks if c.has_table)}")
            print(f"      Chunks with figure refs: {sum(1 for c in chunks if c.has_figure)}")
            
            # Show first chunk preview
            first_chunk = chunks[0]
            text_preview = first_chunk.text[:150] + "..." if len(first_chunk.text) > 150 else first_chunk.text
            print(f"\n   📖 First chunk preview:")
            print(f"      Section: {first_chunk.section or 'N/A'}")
            print(f"      Text: {text_preview}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to parse: {e}")
        logger.exception("Parsing error")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test PDF parsing with EcoloGRAPH")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum number of files to process (default: 5)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        return 1
    
    # Initialize chunker
    chunker = DocumentChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=200
    )
    
    # Collect PDF files
    if input_path.is_file():
        pdf_files = [input_path]
    else:
        pdf_files = list(input_path.glob("*.pdf"))[:args.max_files]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_path}")
        return 1
    
    print(f"\n🔍 Found {len(pdf_files)} PDF file(s) to process")
    
    # Process each PDF
    success_count = 0
    for pdf_file in pdf_files:
        if test_single_pdf(pdf_file, chunker):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary: {success_count}/{len(pdf_files)} PDFs parsed successfully")
    print('='*60)
    
    if success_count == len(pdf_files):
        print("\n🎉 All PDFs parsed successfully! Phase 1 validation complete.")
        return 0
    else:
        print(f"\n⚠️  {len(pdf_files) - success_count} PDF(s) failed to parse.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
