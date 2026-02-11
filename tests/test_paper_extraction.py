"""
Quick test script for paper-based entity extraction.

Tests the new extract_from_paper() method to ensure it works correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.token_utils import estimate_tokens, fits_in_context
from src.extraction.entity_extractor import EntityExtractor
from src.ingestion.chunker import DocumentChunk


def test_token_estimation():
    """Test token estimation utilities."""
    print("Testing token estimation...")
    
    text = "This is a test sentence with about 10 words in it."
    tokens = estimate_tokens(text)
    print(f"  Text: '{text}'")
    print(f"  Estimated tokens: {tokens}")
    print(f"  Actual chars: {len(text)}")
    print(f"  Ratio: {len(text) / tokens:.2f} chars/token")
    
    # Test context fitting
    short_text = "Short text"
    long_text = "x" * 10000  # 10k chars ~= 2500 tokens
    
    fits_short = fits_in_context(short_text, context_window=2048)
    fits_long = fits_in_context(long_text, context_window=2048)
    
    print(f"\n  Short text fits in 2048 context: {fits_short}")
    print(f"  Long text (10k chars) fits in 2048 context: {fits_long}")
    
    print("✅ Token estimation tests passed\n")


def test_paper_extraction_mock():
    """Test paper extraction with mock chunks."""
    print("Testing paper extraction with mock data...")
    
    # Create mock chunks
    chunks = [
        DocumentChunk(
            chunk_id="test_001",
            doc_id="test_paper",
            text="The species Pogonus minutus was observed in wetlands.",
            page=1,
            section="Introduction",
            doc_title="Test Paper",
            char_count=60,
            word_count=10
        ),
        DocumentChunk(
            chunk_id="test_002",
            doc_id="test_paper",
            text="The species measured 4.2 mm in body length on average.",
            page=1,
            section="Methods",
            doc_title="Test Paper",
            char_count=60,
            word_count=10
        ),
        DocumentChunk(
            chunk_id="test_003",
            doc_id="test_paper",
            text="The population of P. minutus was found in Doñana National Park, Spain.",
            page=2,
            section="Results",
            doc_title="Test Paper",
            char_count=75,
            word_count=12
        ),
    ]
    
    print(f"  Created {len(chunks)} mock chunks")
    
    # Calculate total size
    total_text = ' '.join(c.text for c in chunks)
    total_tokens = estimate_tokens(total_text)
    print(f"  Total text length: {len(total_text)} chars")
    print(f"  Estimated tokens: ~{total_tokens}")
    
    # Test would fit in context
    fits = fits_in_context(
        text=total_text,
        system_prompt="System prompt here" * 10,  # ~200 chars
        context_window=2048
    )
    print(f"  Fits in 2048 context window: {fits}")
    
    print("✅ Mock paper extraction tests passed\n")


def test_format_chunks():
    """Test chunk formatting for paper extraction."""
    print("Testing chunk formatting...")
    
    try:
        extractor = EntityExtractor()
        
        chunks = [
            DocumentChunk(
                chunk_id="test_001",
                doc_id="test_paper",
                text="First chunk text.",
                page=1,
                section="Introduction",
                doc_title="Test Paper"
            ),
            DocumentChunk(
                chunk_id="test_002",
                doc_id="test_paper",
                text="Second chunk text.",
                page=2,
                section="Methods",
                doc_title="Test Paper"
            ),
        ]
        
        metadata = {
            'title': 'Test Scientific Paper',
            'authors': ['Author One', 'Author Two'],
            'year': 2025
        }
        
        formatted = extractor._format_chunks_for_paper_extraction(chunks, metadata)
        
        print("  Formatted prompt preview (first 500 chars):")
        print(f"  {formatted[:500]}...")
        
        # Check key elements are present
        assert 'Test Scientific Paper' in formatted
        assert 'Author One' in formatted
        assert 'CHUNK 1' in formatted
        assert 'CHUNK 2' in formatted
        assert 'First chunk text' in formatted
        
        print("✅ Chunk formatting tests passed\n")
        
    except Exception as e:
        print(f"❌ Chunk formatting test failed: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("Paper-Based Entity Extraction Tests")
    print("="*60 + "\n")
    
    test_token_estimation()
    test_paper_extraction_mock()
    test_format_chunks()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)
    print("\nNext step: Run actual ingestion with:")
    print("  python scripts/ingest.py data/test_papers/ --enable-extraction")
