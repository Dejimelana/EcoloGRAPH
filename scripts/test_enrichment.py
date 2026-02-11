"""
Script to test metadata enrichment with real papers.

Usage:
    python scripts/test_enrichment.py --doi "10.1016/j.ecolind.2023.123456"
    python scripts/test_enrichment.py --title "Depth distribution of Atlantic cod"
    python scripts/test_enrichment.py --species "Gadus morhua" "Salmo salar"
"""
import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.enrichment import MetadataEnricher

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_doi_enrichment(enricher: MetadataEnricher, doi: str):
    """Test enrichment by DOI."""
    print(f"\n{'='*60}")
    print(f"🔍 Enriching by DOI: {doi}")
    print('='*60)
    
    result = enricher.enrich_by_doi(doi)
    
    print(f"\n✅ Sources: {', '.join(result.sources) or 'none'}")
    print(f"\n📄 Paper Info:")
    print(f"   Title: {result.title or '(not found)'}")
    print(f"   Year: {result.year or '(not found)'}")
    print(f"   Journal: {result.journal or '(not found)'}")
    print(f"   Publisher: {result.publisher or '(not found)'}")
    
    if result.authors:
        print(f"\n👥 Authors ({len(result.authors)}):")
        for author in result.authors[:5]:
            name = author.get('name', 'Unknown')
            affiliations = author.get('affiliations', [])
            aff_str = f" @ {affiliations[0]}" if affiliations else ""
            print(f"   - {name}{aff_str}")
        if len(result.authors) > 5:
            print(f"   ... and {len(result.authors) - 5} more")
    
    if result.fields_of_study:
        print(f"\n🏷️  Fields of Study: {', '.join(result.fields_of_study)}")
    
    if result.subjects:
        print(f"\n📚 Subjects: {', '.join(result.subjects[:5])}")
    
    if result.funders:
        print(f"\n💰 Funding:")
        for funder in result.funders[:3]:
            print(f"   - {funder.get('name', 'Unknown')}")
    
    print(f"\n📊 Citations: {result.citation_count}")
    print(f"📖 References: {result.reference_count}")
    
    if result.tldr:
        print(f"\n📝 TL;DR: {result.tldr}")
    
    if result.is_open_access:
        print(f"\n🔓 Open Access: {result.open_access_url}")


def test_title_enrichment(enricher: MetadataEnricher, title: str, year: int | None = None):
    """Test enrichment by title."""
    print(f"\n{'='*60}")
    print(f"🔍 Searching by title: {title[:50]}...")
    print('='*60)
    
    result = enricher.enrich_by_title(title, year=year)
    
    if not result.sources:
        print("\n❌ No results found")
        return
    
    print(f"\n✅ Sources: {', '.join(result.sources)}")
    print(f"   DOI: {result.doi or '(not found)'}")
    print(f"   Title: {result.title or '(not found)'}")
    print(f"   Year: {result.year or '(not found)'}")
    print(f"   Journal: {result.journal or '(not found)'}")
    
    if result.fields_of_study:
        print(f"\n🏷️  Fields: {', '.join(result.fields_of_study)}")


def test_species_validation(enricher: MetadataEnricher, species_names: list[str]):
    """Test species name validation."""
    print(f"\n{'='*60}")
    print(f"🔍 Validating {len(species_names)} species names")
    print('='*60)
    
    results = enricher.validate_species(species_names)
    
    for info in results:
        status = "✅" if info.is_valid else "❌"
        source = f"[{info.source}]" if info.source else ""
        marine = "🌊" if info.is_marine else ""
        
        print(f"\n{status} {info.original_name} {source} {marine}")
        
        if info.is_valid:
            print(f"   Scientific name: {info.scientific_name}")
            print(f"   Hierarchy: {info.get_hierarchy_string()}")
            
            if info.is_synonym:
                print(f"   ⚠️  Synonym of: {info.accepted_name}")
            
            if info.aphia_id:
                print(f"   WoRMS ID: {info.aphia_id}")
            if info.gbif_key:
                print(f"   GBIF ID: {info.gbif_key}")


def main():
    parser = argparse.ArgumentParser(description="Test EcoloGRAPH metadata enrichment")
    parser.add_argument("--doi", type=str, help="DOI to enrich")
    parser.add_argument("--title", type=str, help="Title to search")
    parser.add_argument("--year", type=int, help="Publication year (for title search)")
    parser.add_argument("--species", nargs="+", help="Species names to validate")
    parser.add_argument("--cache-dir", type=str, default="data/cache", help="Cache directory")
    
    args = parser.parse_args()
    
    if not any([args.doi, args.title, args.species]):
        # Demo mode
        print("No arguments provided. Running demo...")
        args.doi = "10.1111/faf.12521"  # A real fish ecology paper
        args.species = ["Gadus morhua", "Salmo salar", "Oncorhynchus mykiss"]
    
    # Initialize enricher
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    enricher = MetadataEnricher(cache_dir=cache_dir)
    
    # Run tests
    if args.doi:
        test_doi_enrichment(enricher, args.doi)
    
    if args.title:
        test_title_enrichment(enricher, args.title, args.year)
    
    if args.species:
        test_species_validation(enricher, args.species)
    
    print(f"\n{'='*60}")
    print("✅ Enrichment test complete!")
    print('='*60)


if __name__ == "__main__":
    main()
