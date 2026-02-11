"""
Demo Pipeline Script - Full EcoloGRAPH Pipeline Test

Iteratively processes PDFs to find 5 papers with:
- Same primary domain
- 2-3 overlapping domains (>10% each)

Then runs extraction and shows analysis results.

Usage:
    python scripts/demo_pipeline.py [--input path/to/pdfs]
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pdf_parser import PDFParser, ParsedDocument
from src.ingestion.chunker import DocumentChunker
from src.extraction.domain_classifier import DomainClassifier, ClassificationResult
from src.core.domain_registry import DomainType


@dataclass
class AnalyzedPaper:
    """Paper with classification results."""
    doc_id: str
    path: Path
    title: str
    document: ParsedDocument
    classification: ClassificationResult
    
    @property
    def primary_domain(self) -> DomainType:
        return self.classification.primary_domain
    
    @property
    def domain_scores(self) -> dict[DomainType, float]:
        return self.classification.domain_scores or {}
    
    def get_significant_domains(self, threshold: float = 0.10) -> list[tuple[DomainType, float]]:
        """Get domains with score >= threshold."""
        return [
            (dt, score) for dt, score in self.domain_scores.items()
            if score >= threshold
        ]


def find_overlapping_papers(
    papers: list[AnalyzedPaper],
    min_papers: int = 5,
    overlap_threshold: float = 0.10,
    min_overlapping_domains: int = 2,
    max_overlapping_domains: int = 3
) -> tuple[list[AnalyzedPaper], dict]:
    """
    Find papers that share domains.
    
    Returns:
        Tuple of (matching papers, overlap stats)
    """
    # Group by primary domain
    by_primary: dict[DomainType, list[AnalyzedPaper]] = defaultdict(list)
    for paper in papers:
        by_primary[paper.primary_domain].append(paper)
    
    # Find primary domain with most papers
    best_domain = None
    best_papers = []
    
    for domain, domain_papers in by_primary.items():
        if len(domain_papers) >= min_papers:
            # Check overlap among these papers
            overlap_info = calculate_overlap(
                domain_papers, 
                overlap_threshold,
                min_overlapping_domains,
                max_overlapping_domains
            )
            
            if overlap_info["valid"]:
                best_domain = domain
                best_papers = domain_papers[:min_papers]
                break
    
    if best_papers:
        overlap_stats = calculate_overlap(
            best_papers, 
            overlap_threshold,
            min_overlapping_domains,
            max_overlapping_domains
        )
        return best_papers, overlap_stats
    
    return [], {}


def calculate_overlap(
    papers: list[AnalyzedPaper],
    threshold: float,
    min_domains: int,
    max_domains: int
) -> dict:
    """Calculate domain overlap statistics."""
    # Collect all significant domains across papers
    domain_counts: dict[DomainType, int] = defaultdict(int)
    domain_scores: dict[DomainType, list[float]] = defaultdict(list)
    
    for paper in papers:
        for dt, score in paper.get_significant_domains(threshold):
            domain_counts[dt] += 1
            domain_scores[dt].append(score)
    
    # Find domains present in ALL papers
    n_papers = len(papers)
    shared_domains = [
        (dt, count, sum(domain_scores[dt]) / len(domain_scores[dt]))
        for dt, count in domain_counts.items()
        if count == n_papers
    ]
    
    # Sort by average score
    shared_domains.sort(key=lambda x: x[2], reverse=True)
    
    valid = min_domains <= len(shared_domains) <= max_domains
    
    return {
        "valid": valid,
        "shared_domains": shared_domains,
        "n_shared": len(shared_domains),
        "all_domains": dict(domain_counts),
        "papers_analyzed": n_papers
    }


def display_analysis(papers: list[AnalyzedPaper], overlap_stats: dict):
    """Display detailed analysis of selected papers."""
    
    print("\n" + "=" * 70)
    print("🎯 MATCHED PAPER SET")
    print("=" * 70)
    
    print(f"\n📊 Overlap Statistics:")
    print(f"   Papers: {overlap_stats['papers_analyzed']}")
    print(f"   Shared Domains: {overlap_stats['n_shared']}")
    
    print(f"\n   🔗 Overlapping Domains:")
    for dt, count, avg_score in overlap_stats['shared_domains']:
        bar = "█" * int(avg_score * 20)
        print(f"      • {dt.value:25} {avg_score:6.1%} {bar}")
    
    # Paper details
    print("\n" + "-" * 70)
    print("📑 PAPER DETAILS")
    print("-" * 70)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper.title[:60]}...")
        print(f"    📁 {paper.path.name}")
        print(f"    🏷️  Primary: {paper.primary_domain.value} ({paper.classification.confidence:.1%})")
        print(f"    🧪 Study Type: {paper.classification.study_type.value}")
        
        # Domain vector
        sig_domains = paper.get_significant_domains(0.05)
        if sig_domains:
            print(f"    📊 Domain Vector:")
            for dt, score in sorted(sig_domains, key=lambda x: x[1], reverse=True)[:5]:
                bar = "▓" * int(score * 15)
                print(f"       {dt.value:20} {score:5.1%} {bar}")
    
    # Combined domain vector
    print("\n" + "-" * 70)
    print("🧬 COMBINED DOMAIN VECTOR (Average across papers)")
    print("-" * 70)
    
    combined: dict[DomainType, list[float]] = defaultdict(list)
    for paper in papers:
        for dt, score in paper.domain_scores.items():
            combined[dt].append(score)
    
    # Calculate averages
    avg_vector = {
        dt: sum(scores) / len(scores)
        for dt, scores in combined.items()
    }
    
    # Sort and display
    sorted_vector = sorted(avg_vector.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   Domain                     Avg Score    Variance")
    print("   " + "-" * 50)
    
    for dt, avg in sorted_vector[:10]:
        scores = combined[dt]
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        bar = "█" * int(avg * 30)
        print(f"   {dt.value:25} {avg:6.1%}    ±{variance:.3f} {bar}")
    
    # Output as JSON-compatible dict
    print("\n" + "-" * 70)
    print("📦 VECTOR OUTPUT (for downstream use)")
    print("-" * 70)
    
    output = {
        "papers": [
            {
                "doc_id": p.doc_id,
                "title": p.title,
                "primary_domain": p.primary_domain.value,
                "study_type": p.classification.study_type.value,
                "domain_vector": {dt.value: score for dt, score in p.domain_scores.items()}
            }
            for p in papers
        ],
        "combined_vector": {dt.value: round(score, 4) for dt, score in sorted_vector},
        "shared_domains": [
            {"domain": dt.value, "avg_score": round(avg, 4)}
            for dt, _, avg in overlap_stats['shared_domains']
        ]
    }
    
    print(json.dumps(output, indent=2)[:2000] + "...")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Demo EcoloGRAPH Pipeline")
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Path to PDF directory"
    )
    parser.add_argument(
        "--min-papers", "-n",
        type=int,
        default=5,
        help="Minimum papers to find (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Domain overlap threshold (default: 0.10)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌿 EcoloGRAPH Demo Pipeline")
    print("=" * 70)
    
    # Find PDFs - with interactive fallback
    pdf_dir = Path(args.input)
    
    while not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        if not pdf_dir.exists():
            print(f"\n❌ Directory not found: {pdf_dir}")
        else:
            print(f"\n❌ No PDF files found in: {pdf_dir}")
        
        user_input = input("📁 Enter path to PDF directory (or 'q' to quit): ").strip()
        
        if user_input.lower() == 'q':
            print("Goodbye!")
            return
        
        pdf_dir = Path(user_input)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"\n📁 Found {len(pdf_files)} PDFs in {pdf_dir}")
    
    if len(pdf_files) < args.min_papers:
        print(f"❌ Need at least {args.min_papers} PDFs")
        return
    
    # Initialize components
    print("\n🔧 Initializing components...")
    pdf_parser = PDFParser()
    classifier = DomainClassifier()
    
    # Iteratively process PDFs
    analyzed_papers: list[AnalyzedPaper] = []
    matched_papers: list[AnalyzedPaper] = []
    overlap_stats: dict = {}
    
    print(f"\n🔄 Processing PDFs iteratively (looking for {args.min_papers} overlapping)...\n")
    
    for idx, pdf_path in enumerate(pdf_files):
        print(f"[{idx + 1}/{len(pdf_files)}] Parsing: {pdf_path.name[:50]}...")
        
        try:
            # Parse PDF
            doc = pdf_parser.parse(pdf_path)
            
            # Classify
            result = classifier.classify_document(doc, use_llm=False)
            
            # Create analyzed paper
            paper = AnalyzedPaper(
                doc_id=doc.doc_id,
                path=pdf_path,
                title=doc.title or pdf_path.stem,
                document=doc,
                classification=result
            )
            
            analyzed_papers.append(paper)
            
            # Display quick result
            sig_domains = paper.get_significant_domains(args.threshold)
            domains_str = ", ".join(f"{d.value}({s:.0%})" for d, s in sig_domains[:3])
            print(f"         → {result.primary_domain.value} | {domains_str}")
            
            # Check if we have enough matching papers
            matched_papers, overlap_stats = find_overlapping_papers(
                analyzed_papers,
                min_papers=args.min_papers,
                overlap_threshold=args.threshold,
                min_overlapping_domains=2,
                max_overlapping_domains=3
            )
            
            if matched_papers:
                print(f"\n✅ Found {len(matched_papers)} matching papers!")
                break
                
        except Exception as e:
            print(f"         ⚠️ Error: {e}")
            continue
    
    # Results
    if matched_papers:
        output = display_analysis(matched_papers, overlap_stats)
        
        # Save output
        output_path = Path("data/demo_output.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Output saved to: {output_path}")
        
    else:
        print("\n" + "=" * 70)
        print("⚠️ Could not find enough overlapping papers")
        print("=" * 70)
        
        # Show what we have
        if analyzed_papers:
            print(f"\n📊 Analyzed {len(analyzed_papers)} papers:")
            
            by_domain: dict[DomainType, int] = defaultdict(int)
            for p in analyzed_papers:
                by_domain[p.primary_domain] += 1
            
            print("\n   Primary Domain Distribution:")
            for dt, count in sorted(by_domain.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * count
                print(f"   {dt.value:25} {count:3} {bar}")
            
            print(f"\n💡 Try adding more PDFs or lowering --threshold (current: {args.threshold})")
    
    print("\n" + "=" * 70)
    print("✨ Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
