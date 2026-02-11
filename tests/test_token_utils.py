"""
Simple standalone test for token utilities (no dependencies).
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.token_utils import estimate_tokens, fits_in_context, estimate_safe_batch_size


def test_token_estimation():
    """Test token estimation utilities."""
    print("Testing token estimation...")
    
    # Test 1: Basic estimation
    text = "This is a test sentence with about 10 words in it."
    tokens = estimate_tokens(text)
    expected_range = (10, 20)  # Should be ~13-15 tokens
    
    print(f"  Text: '{text}'")
    print(f"  Length: {len(text)} chars")
    print(f"  Estimated tokens: {tokens}")
    print(f"  Ratio: {len(text) / tokens:.2f} chars/token")
    
    assert expected_range[0] <= tokens <= expected_range[1], \
        f"Token estimate {tokens} outside expected range {expected_range}"
    
    print("  ✅ Basic estimation correct\n")
    
    # Test 2: Context fitting
    short_text = "Short text"
    long_text = "x" * 10000  # 10k chars ~= 2500+ tokens
    
    fits_short = fits_in_context(short_text, context_window=2048)
    fits_long = fits_in_context(long_text, context_window=2048)
    
    print("  Context Fitting:")
    print(f"    Short text (10 chars) fits in 2048 context: {fits_short}")
    print(f"    Long text (10k chars) fits in 2048 context: {fits_long}")
    
    assert fits_short == True, "Short text should fit"
    assert fits_long == False, "Long text should NOT fit"
    
    print("  ✅ Context fitting works correctly\n")


def test_batch_size_estimation():
    """Test batch size estimation."""
    print("Testing batch size estimation...")
    
    # Create mock chunks
    chunk_texts = [f"This is chunk {i} with some test content." for i in range(100)]
    
    batch_size = estimate_safe_batch_size(
        chunk_texts=chunk_texts,
        system_prompt="System prompt" * 20,  # ~200 chars
        max_output_tokens=500,
        context_window=2048,
        template_overhead=200
    )
    
    print(f"  Chunks: {len(chunk_texts)}")
    print(f"  Chunk size: ~{len(chunk_texts[0])} chars each")
    print(f"  Context window: 2048 tokens")
    print(f"  Estimated safe batch size: {batch_size} chunks")
    
    # Should be able to fit at least 1 chunk
    assert batch_size >= 1, "Should fit at least 1 chunk"
    
    # Should not exceed total chunks
    assert batch_size <= len(chunk_texts), "Batch size should not exceed total chunks"
    
    print(f"  ✅ Batch size calculation correct\n")


def test_paper_size_scenarios():
    """Test token estimation for realistic paper scenarios."""
    print("Testing realistic paper scenarios...")
    
    # Scenario 1: Small paper (20 pages, ~50k chars)
    small_paper = "x" * 50000
    small_tokens = estimate_tokens(small_paper)
    small_fits = fits_in_context(small_paper, context_window=2048, max_output_tokens=500)
    
    print("  Scenario 1: Small paper (20 pages)")
    print(f"    Size: {len(small_paper)} chars")
    print(f"    Est. tokens: ~{small_tokens}")
    print(f"    Fits in 2048 context: {small_fits}")
    
    # Scenario 2: Large paper (40 pages, ~120k chars)
    large_paper = "x" * 120000
    large_tokens = estimate_tokens(large_paper)
    large_fits = fits_in_context(large_paper, context_window=2048, max_output_tokens=500)
    
    print("  Scenario 2: Large paper (40 pages)")
    print(f"    Size: {len(large_paper)} chars")
    print(f"    Est. tokens: ~{large_tokens}")
    print(f"    Fits in 2048 context: {large_fits}")
    
    # Calculate how many batches needed
    if not large_fits:
        chunk_size = 1000  # chars per chunk
        num_chunks = len(large_paper) // chunk_size
        chunks = [large_paper[i:i+chunk_size] for i in range(0, len(large_paper), chunk_size)]
        batch_size = estimate_safe_batch_size(
            chunk_texts=[c for c in chunks[:10]],  # Test with first 10
            context_window=2048,
            max_output_tokens=500
        )
        num_batches = (num_chunks // batch_size) + (1 if num_chunks % batch_size else 0)
        
        print(f"    Needs batching: ~{num_chunks} chunks → ~{num_batches} batches")
    
    print("  ✅ Paper scenario estimates reasonable\n")


if __name__ == "__main__":
    print("="*60)
    print("Token Utils Standalone Tests")
    print("="*60 + "\n")
    
    try:
        test_token_estimation()
        test_batch_size_estimation()
        test_paper_size_scenarios()
        
        print("="*60)
        print("✅ All tests PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
