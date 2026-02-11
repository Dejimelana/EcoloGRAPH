"""
Token estimation utilities for LLM context management.

Provides simple heuristics for estimating token counts to prevent
context window overflow.
"""
import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using simple character-based heuristic.
    
    Rule of thumb: 1 token ≈ 4 characters for English text.
    This is conservative and works reasonably well for scientific papers.
    
    Args:
        text: Input text to estimate
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Base estimate: 1 token per 4 chars
    base_tokens = len(text) // 4
    
    # Add safety margin (10% or minimum 10 tokens)
    safety_margin = max(10, int(base_tokens * 0.1))
    
    return base_tokens + safety_margin


def fits_in_context(
    text: str,
    system_prompt: str = "",
    max_output_tokens: int = 500,
    context_window: int = 2048
) -> bool:
    """
    Check if text + prompt will fit in model's context window.
    
    Args:
        text: Main text content
        system_prompt: System prompt to include
        max_output_tokens: Expected max output tokens
        context_window: Model's context window size
        
    Returns:
        True if fits, False if would overflow
    """
    input_tokens = estimate_tokens(system_prompt + text)
    total_tokens = input_tokens + max_output_tokens
    
    fits = total_tokens <= context_window
    
    if not fits:
        logger.debug(
            f"Context check: {total_tokens} tokens > {context_window} window "
            f"(input: {input_tokens}, output: {max_output_tokens})"
        )
    
    return fits


def estimate_safe_batch_size(
    chunk_texts: list[str],
    system_prompt: str = "",
    max_output_tokens: int = 500,
    context_window: int = 2048,
    template_overhead: int = 200
) -> int:
    """
    Calculate how many chunks can fit in one batch.
    
    Args:
        chunk_texts: List of chunk text contents
        system_prompt: System prompt (counted once)
        max_output_tokens: Reserve for output
        context_window: Model's context window
        template_overhead: Tokens for batch template formatting
        
    Returns:
        Number of chunks that fit (at least 1)
    """
    base_tokens = estimate_tokens(system_prompt) + template_overhead + max_output_tokens
    available_tokens = context_window - base_tokens
    
    if available_tokens <= 0:
        logger.warning(f"System prompt + overhead leaves no room for content!")
        return 1  # Try at least 1 chunk
    
    # Calculate cumulative tokens
    cumulative = 0
    for i, chunk_text in enumerate(chunk_texts):
        chunk_tokens = estimate_tokens(chunk_text)
        chunk_tokens += 50  # Metadata overhead per chunk
        
        if cumulative + chunk_tokens > available_tokens:
            # This chunk would overflow
            return max(1, i)  # Return previous count (at least 1)
        
        cumulative += chunk_tokens
    
    # All chunks fit
    return len(chunk_texts)
