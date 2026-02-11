"""
Script to test LLM extraction with real document chunks.

Usage:
    python scripts/test_extraction.py --input "data/processed/sample.json"
    python scripts/test_extraction.py --demo  # Run with sample text
"""
import argparse
import sys
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Sample ecological text for demo
SAMPLE_TEXT = """
In the North Sea, Atlantic cod (Gadus morhua) populations have shown significant 
changes in depth distribution over the past two decades. Adult cod were found primarily 
at depths between 50 and 150 meters during winter months (December-February), with 
mean depth increasing from 75m in 2000 to 110m in 2020.

Juveniles (ages 1-2) were typically observed in shallower coastal waters, at depths 
of 20-40 meters, where they feed on small crustaceans and juvenile herring (Clupea harengus).

Temperature measurements showed a mean bottom temperature of 7.2°C (±0.8°C, n=156) 
in the study area, with values ranging from 5.1°C to 9.4°C across seasons.

The predation relationship between cod and herring is well documented, with cod 
consuming approximately 15% of juvenile herring biomass annually in this region.
"""


def test_llm_availability(llm: LLMClient) -> bool:
    """Check if LLM is available."""
    print("\n" + "="*60)
    print("🔌 Checking LLM availability...")
    print("="*60)
    
    # For LM Studio, we can't check /v1/models - just test connection
    try:
        import httpx
        response = httpx.get(f"{llm.base_url.rstrip('/v1')}", timeout=5.0)
        # LM Studio returns something on the root endpoint
        print(f"✅ Connected to {llm.base_url}")
        print(f"   Model: {llm.model} (will load on first request)")
        return True
    except Exception:
        # Try anyway - JIT loading might work
        print(f"⚠️  Cannot verify connection, but will try anyway...")
        print(f"   Base URL: {llm.base_url}")
        return True  # Return true to attempt extraction


def test_extraction_demo(extractor: EntityExtractor):
    """Run extraction on sample text."""
    print("\n" + "="*60)
    print("🔬 Running extraction demo...")
    print("="*60)
    
    # Create a mock chunk
    from src.ingestion.chunker import DocumentChunk
    
    chunk = DocumentChunk(
        chunk_id="demo_chunk_001",
        doc_id="demo_doc",
        text=SAMPLE_TEXT,
        section="Results",
        page=1,
        chunk_idx=0,
        word_count=len(SAMPLE_TEXT.split()),
        char_count=len(SAMPLE_TEXT)
    )
    
    print(f"\n📄 Processing demo text ({chunk.word_count} words)...")
    print(f"   Section: {chunk.section}")
    
    # Extract
    result = extractor.extract_from_chunk(chunk)
    
    # Display results
    print(f"\n✅ Extraction complete!")
    print(f"   Total entities: {result.entity_count()}")
    print(f"   Model used: {result.model_used}")
    
    if result.species:
        print(f"\n🐟 Species ({len(result.species)}):")
        for sp in result.species:
            print(f"   - {sp.original_name}")
            if sp.source and sp.source.text_snippet:
                print(f"     📝 \"{sp.source.text_snippet[:60]}...\"")
    
    if result.measurements:
        print(f"\n📏 Measurements ({len(result.measurements)}):")
        for m in result.measurements:
            value_str = f"{m.value}"
            if m.value_min and m.value_max:
                value_str = f"{m.value_min}-{m.value_max}"
            print(f"   - {m.parameter}: {value_str} {m.unit}")
            if m.species:
                print(f"     (for: {m.species})")
            if m.source and m.source.text_snippet:
                print(f"     📝 \"{m.source.text_snippet[:60]}...\"")
    
    if result.locations:
        print(f"\n📍 Locations ({len(result.locations)}):")
        for loc in result.locations:
            info = loc.name
            if loc.region:
                info += f" ({loc.region})"
            print(f"   - {info}")
    
    if result.temporal_info:
        print(f"\n📅 Temporal ({len(result.temporal_info)}):")
        for t in result.temporal_info:
            info = t.description or f"{t.start_date} to {t.end_date}"
            if t.season:
                info += f" ({t.season})"
            print(f"   - {info}")
    
    if result.relations:
        print(f"\n🔗 Relations ({len(result.relations)}):")
        for r in result.relations:
            print(f"   - {r.subject} [{r.relation_type.value}] {r.object}")
            if r.description:
                print(f"     ({r.description})")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test EcoloGRAPH LLM extraction")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample text")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1", 
                        help="LLM API base URL")
    parser.add_argument("--api-key", type=str, help="API key for LLM server")
    
    args = parser.parse_args()
    
    # Initialize LLM client
    llm = LLMClient(
        model=args.model,
        base_url=args.base_url,
        temperature=0.1,
        api_key=args.api_key
    )
    
    # Check availability
    if not test_llm_availability(llm):
        print("\n⚠️  Skipping extraction test (LLM not available)")
        return
    
    # Initialize extractor
    extractor = EntityExtractor(llm_client=llm)
    
    # Run demo
    if args.demo or True:  # Always run demo for now
        test_extraction_demo(extractor)
    
    print("\n" + "="*60)
    print("✅ Extraction test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
