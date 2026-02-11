"""
Fix paper titles, years, and abstracts in the SQLite database.

Uses PyMuPDF to extract text from PDFs with font-size analysis:
- Title = largest text on page 1 (scientific papers always use the biggest font for title)
- Year = first 4-digit year in the first page text
- Abstract = text between "Abstract" and "Introduction"/"Keywords" on first pages

Usage:
    python scripts/fix_paper_metadata.py                 # Fix all
    python scripts/fix_paper_metadata.py --dry-run       # Preview
    python scripts/fix_paper_metadata.py --skip-pdf      # Filename fallback only
    python scripts/fix_paper_metadata.py --fix-abstracts  # Also fix bad abstracts
"""
import re
import sqlite3
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path("data/paper_index.db")


def extract_from_pdf(pdf_path: str) -> dict:
    """
    Extract title, year, and abstract from PDF using PyMuPDF font-size analysis.
    
    Strategy for title: the largest font on page 1 is almost always the paper title.
    This is far more reliable than text-position heuristics.
    """
    import pymupdf  # PyMuPDF
    
    result = {"title": None, "year": None, "abstract": None}
    
    p = Path(pdf_path)
    if not p.exists():
        return result
    
    try:
        doc = pymupdf.open(str(p))
        if doc.page_count == 0:
            doc.close()
            return result
        
        # --- TITLE: Find largest font text on page 1 ---
        page = doc[0]
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        # Collect text spans with their font sizes
        text_by_size = defaultdict(list)
        all_spans = []
        
        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text and size > 0:
                        all_spans.append((size, text))
                        text_by_size[round(size, 1)].append(text)
        
        max_size = 14.0  # default title font size
        if all_spans:
            # Find the largest font size (excluding very short fragments)
            # Group by font size and find the largest group with substantial text
            max_size = max(s for s, t in all_spans if len(t) > 3)
            
            # Collect all text at that size (title may span multiple lines)
            title_parts = []
            for size, text in all_spans:
                if abs(size - max_size) < 0.5 and len(text) > 2:
                    # Skip purely numeric fragments, page headers, etc.
                    if not re.match(r'^[\d\s.,;:]+$', text):
                        title_parts.append(text)
            
            if title_parts:
                title = " ".join(title_parts).strip()
                # Clean up common artifacts
                title = re.sub(r'\s+', ' ', title)
                # Remove trailing/leading punctuation artifacts
                title = title.strip('.,;: ')
                if len(title) > 10 and len(title) < 500:
                    result["title"] = title
        
        # --- YEAR: Find first 4-digit year in first page ---
        first_page_text = page.get_text()
        year_matches = re.findall(r'\b(20[0-2]\d|19[89]\d)\b', first_page_text)
        if year_matches:
            result["year"] = int(year_matches[0])
        
        # --- ABSTRACT: Find text between "Abstract" and next section ---
        # Combine first 2-3 pages of text
        pages_text = ""
        for i in range(min(3, doc.page_count)):
            pages_text += doc[i].get_text() + "\n"
        
        # Strategy 1: "Abstract" followed by text, ending at Keywords/Introduction/1.
        abs_match = re.search(
            r'[Aa]bstract\.?\s*\n?\s*(.*?)(?=\s*(?:Keywords?|Introduction|INTRODUCTION|1\.\s|©|\bDOI\b))',
            pages_text, re.DOTALL
        )
        if abs_match:
            abstract = abs_match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            if 50 < len(abstract) < 3000:
                result["abstract"] = abstract
        
        # Strategy 2: "ABSTRACT" in all caps
        if not result["abstract"]:
            abs_match2 = re.search(
                r'ABSTRACT\s*\n?\s*(.*?)(?=\s*(?:KEYWORDS?|INTRODUCTION|1\.\s))',
                pages_text, re.DOTALL
            )
            if abs_match2:
                abstract = re.sub(r'\s+', ' ', abs_match2.group(1).strip())
                if 50 < len(abstract) < 3000:
                    result["abstract"] = abstract
        
        # Strategy 3: No "Abstract" heading (ScienceDirect, etc.)
        # Use block layout: abstract is the longest regular-font text block on page 1
        if not result["abstract"]:
            blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
            
            # Get title font size to exclude title blocks
            title_size = max_size
            
            # Find the longest text block that isn't the title/metadata
            candidate_blocks = []
            for block in blocks:
                if block.get("type") != 0:
                    continue
                
                text = ""
                font_sizes = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "") + " "
                        font_sizes.append(span.get("size", 0))
                
                text = re.sub(r'\s+', ' ', text).strip()
                avg_size = sum(font_sizes)/len(font_sizes) if font_sizes else 0
                
                # Skip very short blocks, title-sized blocks, tiny metadata font
                if len(text) < 100:
                    continue
                if avg_size > title_size - 1:  # skip title blocks
                    continue
                if avg_size < 5.5:  # skip very small metadata text
                    continue
                # Skip metadata patterns
                if re.match(r'^(Available|Received|Accepted|Published|http|www\.|©|doi|Contents)', text, re.IGNORECASE):
                    continue
                if text.count('@') > 0:  # email addresses (affiliations)
                    continue
                
                candidate_blocks.append((len(text), text))
            
            # The abstract is typically the longest qualifying block
            if candidate_blocks:
                candidate_blocks.sort(key=lambda x: x[0], reverse=True)
                best = candidate_blocks[0][1]
                if 80 < len(best) < 3000:
                    result["abstract"] = best
        
        doc.close()
        
    except Exception as e:
        print(f"    ⚠️  PDF error: {e}")
    
    return result


def title_from_filename(filename: str) -> tuple[str, int | None]:
    """
    Extract title and year from PDF filename (last resort).
    ScienceDirect: Title-words_YEAR_Journal.pdf
    """
    stem = Path(filename).stem
    year = None
    title = stem
    
    match = re.search(r'_(\d{4})_', stem)
    if match:
        year = int(match.group(1))
        title = stem[:match.start()]
    
    title = re.sub(r'---+', ' — ', title)
    title = re.sub(r'--', ' – ', title)
    title = title.replace('-', ' ').replace('_', ' ')
    title = re.sub(r'\s+', ' ', title).strip()
    if title:
        title = title[0].upper() + title[1:]
    title = re.sub(r'\s+\w$', '', title)
    
    return title, year


def resolve_source_path(source_path: str, project_root: Path) -> str:
    """Resolve a relative source path to an absolute path."""
    if not source_path:
        return source_path
    
    p = Path(source_path)
    
    if p.is_absolute() and p.exists():
        return str(p)
    
    # Try relative to project root
    resolved = (project_root / p).resolve()
    if resolved.exists():
        return str(resolved)
    
    # Try relative to CWD
    resolved = (Path.cwd() / p).resolve()
    if resolved.exists():
        return str(resolved)
    
    # Search common locations
    fname = p.name
    for search_dir in [
        project_root / "data" / "raw",
        project_root.parent / "scientific-rag-assistant" / "data" / "raw",
    ]:
        candidate = search_dir / fname
        if candidate.exists():
            return str(candidate.resolve())
    
    return source_path


def main():
    parser = argparse.ArgumentParser(description="Fix paper metadata in SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF content extraction (faster)")
    parser.add_argument("--fix-abstracts", action="store_true", help="Also re-extract abstracts")
    args = parser.parse_args()
    
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    project_root = Path(__file__).parent.parent.resolve()
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT doc_id, title, year, source_path, abstract FROM papers").fetchall()
    
    print(f"📊 Found {len(rows)} papers in database")
    
    updated = 0
    for i, row in enumerate(rows, 1):
        doc_id = row["doc_id"]
        old_title = row["title"]
        old_year = row["year"]
        old_path = row["source_path"]
        old_abstract = row["abstract"]
        
        # Detect bad data
        bad_title = (
            not old_title 
            or old_title == "Untitled" 
            or old_title.startswith("<!--")
            or len(old_title) < 5
        )
        needs_year = old_year is None
        bad_abstract = args.fix_abstracts and (
            not old_abstract 
            or 'abstractCls' in (old_abstract or '')
            or old_abstract.startswith('abstract')  # raw "abstract..." prefix
            or len(old_abstract or '') < 30
        )
        
        if not bad_title and not needs_year and not bad_abstract:
            continue
        
        changes = {}
        
        # Resolve path first
        resolved_path = resolve_source_path(old_path, project_root) if old_path else old_path
        if resolved_path != old_path:
            changes["source_path"] = resolved_path
        
        # Strategy 1: PyMuPDF extraction
        if not args.skip_pdf:
            actual_path = resolved_path or old_path
            if actual_path and Path(actual_path).exists():
                print(f"  [{i}/{len(rows)}] 📄 {Path(actual_path).name[:50]}...", end=" ", flush=True)
                extracted = extract_from_pdf(actual_path)
                
                if bad_title and extracted["title"]:
                    changes["title"] = extracted["title"]
                    print(f"✅ \"{extracted['title'][:50]}...\"")
                elif bad_title:
                    print("⚠️ no title from PDF", end=" ")
                else:
                    print("", end="")
                
                if needs_year and extracted["year"]:
                    changes["year"] = extracted["year"]
                
                if bad_abstract and extracted["abstract"]:
                    changes["abstract"] = extracted["abstract"]
                    print(f" 📝 abstract: {len(extracted['abstract'])} chars")
                else:
                    print()
            else:
                print(f"  [{i}/{len(rows)}] ❌ PDF not found: {old_path}")
        
        # Strategy 2: Filename fallback for title/year
        if "title" not in changes and bad_title and old_path:
            fn_title, fn_year = title_from_filename(Path(old_path).name)
            if fn_title:
                changes["title"] = fn_title
            if fn_year and needs_year and "year" not in changes:
                changes["year"] = fn_year
            if "title" in changes:
                print(f"  [{i}/{len(rows)}] 📁 Filename: \"{changes['title'][:50]}...\" ({changes.get('year', '?')})")
        
        # Apply changes
        if changes and not args.dry_run:
            set_clauses = ", ".join(f"{k} = ?" for k in changes.keys())
            values = list(changes.values()) + [doc_id]
            conn.execute(f"UPDATE papers SET {set_clauses} WHERE doc_id = ?", values)
            updated += 1
        elif changes:
            updated += 1  # dry-run count
    
    if not args.dry_run:
        conn.commit()
    conn.close()
    
    mode = " (DRY RUN)" if args.dry_run else ""
    print(f"\n📊 Updated {updated}/{len(rows)} papers{mode}")
    if not args.dry_run and updated > 0:
        print("✅ Done! Restart Streamlit to see changes.")
    elif args.dry_run:
        print("💡 Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
