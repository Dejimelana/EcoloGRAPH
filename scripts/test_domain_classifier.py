"""
Test script for domain classification.

Usage:
    python scripts/test_domain_classifier.py --input "path/to/pdfs"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.llm_client import LLMClient
from src.core.domain_registry import DomainRegistry, DomainType
from src.extraction.domain_classifier import DomainClassifier
from src.ingestion.pdf_parser import PDFParser


def test_classifier(input_path: str, max_files: int = 5, api_key: str = None):
    """Test domain classification on PDFs."""
    
    print("\n" + "="*70)
    print("🏷️  Domain Classification Test")
    print("="*70)
    
    # List available domains
    print("\n📋 Registered domains:")
    for domain in DomainRegistry.get_all():
        print(f"   - {domain.domain_type.value}: {domain.display_name}")
    
    # Find PDFs
    input_dir = Path(input_path)
    pdf_files = list(input_dir.glob("*.pdf"))[:max_files]
    print(f"\n📁 Found {len(pdf_files)} PDFs to classify")
    
    # Initialize
    parser = PDFParser()
    llm = LLMClient(api_key=api_key) if api_key else None
    classifier = DomainClassifier(llm_client=llm)
    
    if llm:
        print(f"   Using LLM: {llm.model}")
    else:
        print("   Using keyword-only classification")
    
    # Classify each PDF
    results = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*70}")
        print(f"📄 [{i}/{len(pdf_files)}] {pdf_path.name[:60]}...")
        print("="*70)
        
        # Parse
        doc = parser.parse(pdf_path)
        if not doc:
            print("   ❌ Failed to parse")
            continue
        
        # Classify
        result = classifier.classify_document(doc, use_llm=bool(llm))
        
        # Display
        config = classifier.get_domain_config(result.primary_domain)
        
        print(f"\n   🏷️  Domain: {config.display_name if config else result.primary_domain.value}")
        print(f"   📊 Confidence: {result.confidence:.2%}")
        print(f"   🔧 Method: {result.method}")
        print(f"   🧪 Study Type: {result.study_type.value} ({result.study_type_confidence:.2%})")
        print(f"   💭 Reasoning: {result.reasoning}")
        
        # Multi-label scores
        if result.domain_scores:
            top_domains = classifier.get_top_domains(result, threshold=0.05)
            if len(top_domains) > 1:
                print(f"\n   📊 Multi-Domain Scores:")
                for domain, score in top_domains[:5]:
                    marker = "★" if domain == result.primary_domain else " "
                    print(f"      {marker} {domain.value}: {score:.2%}")
        
        if result.secondary_domains:
            print(f"\n   📌 Secondary domains:")
            for domain, score in result.secondary_domains[:2]:
                print(f"      - {domain.value}: {score:.2%}")
        
        if config:
            print(f"\n   🔌 Enrichment APIs: {', '.join(config.enrichment_apis)}")
            print(f"   📦 Entity types: {', '.join(config.entity_types)}")
        
        results.append({
            "file": pdf_path.name,
            "domain": result.primary_domain.value,
            "study_type": result.study_type.value,
            "confidence": result.confidence
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Classification Summary")
    print("="*70)
    
    domain_counts = {}
    for r in results:
        domain = r["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"   {domain}: {count} documents")
    
    print(f"\n✅ Classified {len(results)} documents")


def main():
    parser = argparse.ArgumentParser(description="Test domain classification")
    parser.add_argument("--input", type=str,
                        default=r"c:\Users\Usuario\Documents\AntiGravity Projects\Bootcamp\scientific-rag-assistant\data\raw",
                        help="Path to PDF directory")
    parser.add_argument("--max-files", type=int, default=5, help="Max PDFs")
    parser.add_argument("--api-key", type=str, help="LLM API key")
    
    args = parser.parse_args()
    test_classifier(args.input, args.max_files, args.api_key)


if __name__ == "__main__":
    main()
