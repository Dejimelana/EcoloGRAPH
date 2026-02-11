"""
Full pipeline test: Parse PDFs → Chunk → Extract with LLM.

Usage:
    python scripts/test_full_pipeline.py --input "path/to/pdfs" --max-files 5
"""
import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.core.llm_client import LLMClient
from src.extraction.entity_extractor import EntityExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(input_path: str, max_files: int = 5, api_key: str = None):
    """Run full extraction pipeline on PDFs."""
    
    print("\n" + "="*70)
    print("🔬 EcoloGRAPH Full Pipeline Test")
    print("="*70)
    
    # Find PDFs
    input_dir = Path(input_path)
    if not input_dir.exists():
        print(f"❌ Directory not found: {input_dir}")
        return
    
    pdf_files = list(input_dir.glob("*.pdf"))[:max_files]
    print(f"\n📁 Found {len(pdf_files)} PDFs to process")
    
    # Initialize components
    print("\n🔧 Initializing pipeline components...")
    parser = PDFParser()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    llm = LLMClient(api_key=api_key)
    extractor = EntityExtractor(llm_client=llm)
    
    print(f"   Model: {llm.model}")
    
    # Process each PDF
    all_results = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*70}")
        print(f"📄 [{i}/{len(pdf_files)}] {pdf_path.name}")
        print("="*70)
        
        # Parse
        print("   📖 Parsing PDF...")
        doc = parser.parse(pdf_path)
        if not doc:
            print("   ❌ Failed to parse")
            continue
        
        print(f"   ✅ Parsed: {doc.title or 'Unknown title'}")
        print(f"      Authors: {', '.join(doc.authors[:2])}{'...' if len(doc.authors) > 2 else ''}")
        print(f"      Sections: {len(doc.sections)}")
        
        # Chunk
        print("   📦 Chunking...")
        chunks = chunker.chunk_document(doc)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Extract from first few chunks (limit for demo)
        max_chunks = 3
        print(f"   🔍 Extracting entities from {min(len(chunks), max_chunks)} chunks...")
        
        doc_entities = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "species": [],
            "measurements": [],
            "locations": [],
            "relations": []
        }
        
        for j, chunk in enumerate(chunks[:max_chunks]):
            result = extractor.extract_from_chunk(chunk)
            
            # Aggregate
            for sp in result.species:
                if sp.original_name not in [s["name"] for s in doc_entities["species"]]:
                    doc_entities["species"].append({
                        "name": sp.original_name,
                        "section": chunk.section
                    })
            
            for m in result.measurements:
                doc_entities["measurements"].append({
                    "parameter": m.parameter,
                    "value": m.value,
                    "unit": m.unit,
                    "species": m.species,
                    "section": chunk.section
                })
            
            for loc in result.locations:
                if loc.name not in [l["name"] for l in doc_entities["locations"]]:
                    doc_entities["locations"].append({
                        "name": loc.name,
                        "region": loc.region
                    })
            
            for rel in result.relations:
                doc_entities["relations"].append({
                    "type": rel.relation_type.value,
                    "subject": rel.subject,
                    "object": rel.object
                })
        
        # Print summary for this document
        print(f"\n   📊 Extraction Results:")
        print(f"      🐟 Species: {len(doc_entities['species'])}")
        for sp in doc_entities['species'][:5]:
            print(f"         - {sp['name']} (in: {sp['section']})")
        
        print(f"      📏 Measurements: {len(doc_entities['measurements'])}")
        for m in doc_entities['measurements'][:3]:
            sp_str = f" ({m['species']})" if m['species'] else ""
            print(f"         - {m['parameter']}: {m['value']} {m['unit']}{sp_str}")
        
        print(f"      📍 Locations: {len(doc_entities['locations'])}")
        for loc in doc_entities['locations'][:3]:
            print(f"         - {loc['name']}")
        
        print(f"      🔗 Relations: {len(doc_entities['relations'])}")
        for rel in doc_entities['relations'][:3]:
            print(f"         - {rel['subject']} [{rel['type']}] {rel['object']}")
        
        all_results.append(doc_entities)
    
    # Save results
    output_file = Path("data/processed/extraction_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": llm.model,
            "documents": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Pipeline complete! Results saved to: {output_file}")
    print("="*70)
    
    # Summary
    total_species = sum(len(d["species"]) for d in all_results)
    total_measurements = sum(len(d["measurements"]) for d in all_results)
    total_relations = sum(len(d["relations"]) for d in all_results)
    
    print(f"\n📈 Total Extraction Summary:")
    print(f"   Documents processed: {len(all_results)}")
    print(f"   Unique species: {total_species}")
    print(f"   Measurements: {total_measurements}")
    print(f"   Relations: {total_relations}")


def main():
    parser = argparse.ArgumentParser(description="Test full EcoloGRAPH pipeline")
    parser.add_argument("--input", type=str, 
                        default=r"c:\Users\Usuario\Documents\AntiGravity Projects\Bootcamp\scientific-rag-assistant\data\raw",
                        help="Path to PDF directory")
    parser.add_argument("--max-files", type=int, default=5, help="Max PDFs to process")
    parser.add_argument("--api-key", type=str, help="LLM API key")
    
    args = parser.parse_args()
    run_pipeline(args.input, args.max_files, args.api_key)


if __name__ == "__main__":
    main()
