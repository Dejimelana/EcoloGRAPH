#!/usr/bin/env python3
"""
Backfill SQLite chunks table from existing PDFs using PyMuPDF.

Extracts text from PDFs, chunks it, and stores in the SQLite `chunks` table
so chunks are available in the UI without Qdrant.

Usage:
    python scripts/backfill_chunks.py
"""
import sys
import sqlite3
import logging
import hashlib
import re
import argparse
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000    # characters
CHUNK_OVERLAP = 200
MIN_CHUNK = 100


def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from PDF, returning list of (page_num, text)."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append((page.number + 1, text.strip()))
    doc.close()
    return pages


def chunk_text(doc_id: str, pages: list[tuple[int, str]]) -> list[dict]:
    """Split page text into overlapping chunks."""
    # Join all pages with page markers
    full_text = "\n\n".join(text for _, text in pages)
    if not full_text.strip():
        return []

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(full_text):
        end = min(start + CHUNK_SIZE, len(full_text))

        # Try to break at sentence boundary
        if end < len(full_text):
            search_start = max(start, end - 100)
            search_end = min(len(full_text), end + 100)
            snippet = full_text[search_start:search_end]
            for sep in ['. ', '? ', '! ', '.\n', '?\n', '!\n']:
                pos = snippet.rfind(sep)
                if pos > 0:
                    end = search_start + pos + 2
                    break

        chunk_text_str = full_text[start:end].strip()

        if len(chunk_text_str) >= MIN_CHUNK:
            # Determine page from character offset
            page_num = None
            char_count = 0
            for pg, pg_text in pages:
                char_count += len(pg_text) + 2  # +2 for \n\n join
                if char_count >= start:
                    page_num = pg
                    break

            # Detect section heading (line starting with uppercase, < 80 chars)
            section = None
            lines = chunk_text_str.split('\n')
            for line in lines[:3]:
                line = line.strip()
                if line and len(line) < 80 and line[0].isupper() and not line.endswith('.'):
                    section = line
                    break

            chunk_id = hashlib.md5(
                f"{doc_id}:{chunk_idx}:{chunk_text_str[:50]}".encode()
            ).hexdigest()[:16]

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text_str,
                "section": section,
                "page": page_num,
                "chunk_idx": chunk_idx,
                "word_count": len(chunk_text_str.split()),
            })
            chunk_idx += 1

        # Ensure forward progress (avoid infinite loop)
        new_start = end - CHUNK_OVERLAP
        if new_start <= start:
            new_start = start + max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
        start = new_start

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Backfill SQLite chunks from PDFs")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0=all)")
    args = parser.parse_args()

    db_path = ROOT / "data" / "paper_index.db"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return

    # Ensure chunks table exists
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                section TEXT,
                page INTEGER,
                chunk_idx INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                FOREIGN KEY (doc_id) REFERENCES papers(doc_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_order ON chunks(doc_id, chunk_idx)")
        conn.commit()

    # Get all papers
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT doc_id, title, source_path FROM papers WHERE source_path IS NOT NULL"
        ).fetchall()

    # Already chunked
    with sqlite3.connect(str(db_path)) as conn:
        chunked = {r[0] for r in conn.execute(
            "SELECT DISTINCT doc_id FROM chunks"
        ).fetchall()}

    # Apply limit
    to_process = [r for r in rows if r["doc_id"] not in chunked]
    if args.limit > 0:
        to_process = to_process[:args.limit]
    logger.info(f"📋 {len(rows)} papers, {len(chunked)} already chunked, processing {len(to_process)}")

    total_chunks = 0
    skipped = 0
    errors = 0

    for row in to_process:
        doc_id = row["doc_id"]
        title = row["title"] or "Untitled"
        source_path = row["source_path"]

        pdf_path = Path(source_path)
        if not pdf_path.exists():
            logger.warning(f"  ⚠️  Not found: {title[:50]}...")
            errors += 1
            continue

        try:
            pages = extract_text_by_page(pdf_path)
            chunks = chunk_text(doc_id, pages)

            if chunks:
                with sqlite3.connect(str(db_path)) as conn:
                    for c in chunks:
                        conn.execute(
                            """INSERT OR REPLACE INTO chunks
                               (chunk_id, doc_id, text, section, page, chunk_idx, word_count)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (c["chunk_id"], c["doc_id"], c["text"],
                             c["section"], c["page"], c["chunk_idx"],
                             c["word_count"])
                        )
                    conn.commit()
                total_chunks += len(chunks)
                logger.info(f"  ✅ {title[:60]} → {len(chunks)} chunks")
            else:
                logger.warning(f"  ⚠️  0 chunks: {title[:50]}")
        except Exception as e:
            logger.error(f"  ❌ {title[:50]}: {e}")
            errors += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Backfill complete:")
    logger.info(f"   Chunks stored: {total_chunks}")
    logger.info(f"   Skipped (already chunked): {skipped}")
    logger.info(f"   Errors: {errors}")


if __name__ == "__main__":
    main()
