"""
Rebuild the FTS5 index for paper_index.db.

Usage:
    python scripts/fix_fts5.py

Fixes the "fts5: missing row N" error by dropping and rebuilding
the FTS5 virtual table and its triggers.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "paper_index.db"


def rebuild_fts5(db_path: Path = DB_PATH):
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    print(f"Database: {db_path}")

    conn = sqlite3.connect(db_path)

    # Count papers
    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"Papers in table: {count}")

    # Step 1: Drop triggers
    print("\nStep 1: Dropping old triggers...")
    for trigger in ["papers_ai", "papers_ad", "papers_au"]:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.commit()
    print("   OK Triggers dropped")

    # Step 2: Drop old FTS5 table
    print("Step 2: Dropping old FTS5 table...")
    conn.execute("DROP TABLE IF EXISTS papers_fts")
    conn.commit()
    print("   OK FTS5 table dropped")

    # Step 3: Recreate FTS5 table
    print("Step 3: Creating new FTS5 table...")
    conn.execute("""
        CREATE VIRTUAL TABLE papers_fts USING fts5(
            doc_id,
            title,
            abstract,
            keywords,
            authors,
            content='papers',
            content_rowid='rowid'
        )
    """)
    conn.commit()
    print("   OK FTS5 table created")

    # Step 4: Populate FTS5 from content table
    print("Step 4: Rebuilding FTS5 index...")
    conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    conn.commit()
    print("   OK FTS5 index rebuilt")

    # Step 5: Recreate triggers
    print("Step 5: Recreating sync triggers...")
    conn.execute("""
        CREATE TRIGGER papers_ai AFTER INSERT ON papers BEGIN
            INSERT INTO papers_fts(rowid, doc_id, title, abstract, keywords, authors)
            VALUES (new.rowid, new.doc_id, new.title, new.abstract, new.keywords, new.authors);
        END
    """)
    conn.execute("""
        CREATE TRIGGER papers_ad AFTER DELETE ON papers BEGIN
            INSERT INTO papers_fts(papers_fts, rowid, doc_id, title, abstract, keywords, authors)
            VALUES ('delete', old.rowid, old.doc_id, old.title, old.abstract, old.keywords, old.authors);
        END
    """)
    conn.execute("""
        CREATE TRIGGER papers_au AFTER UPDATE ON papers BEGIN
            INSERT INTO papers_fts(papers_fts, rowid, doc_id, title, abstract, keywords, authors)
            VALUES ('delete', old.rowid, old.doc_id, old.title, old.abstract, old.keywords, old.authors);
            INSERT INTO papers_fts(rowid, doc_id, title, abstract, keywords, authors)
            VALUES (new.rowid, new.doc_id, new.title, new.abstract, new.keywords, new.authors);
        END
    """)
    conn.commit()
    print("   OK Triggers recreated")

    # Step 6: Verify
    print("\nVerifying...")
    try:
        results = conn.execute(
            "SELECT doc_id, title FROM papers_fts WHERE papers_fts MATCH 'ecology' LIMIT 3"
        ).fetchall()
        print(f"   OK Test search 'ecology' returned {len(results)} results")
        for r in results:
            print(f"      - {r[1][:60]}...")
    except Exception as e:
        print(f"   FAIL Test search failed: {e}")

    # Integrity check
    try:
        conn.execute(
            "INSERT INTO papers_fts(papers_fts) VALUES('integrity-check')"
        )
        print("   OK FTS5 integrity check passed")
    except Exception as e:
        print(f"   FAIL FTS5 integrity check failed: {e}")

    conn.close()
    print("\nDone! FTS5 index rebuilt successfully.")


if __name__ == "__main__":
    rebuild_fts5()
