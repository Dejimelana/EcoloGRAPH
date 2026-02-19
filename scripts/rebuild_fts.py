"""Rebuild the papers table and FTS5 index properly."""
import sqlite3

db_path = "data/paper_index.db"
print(f"Fixing database at {db_path}...")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check current state
tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"  Current tables: {tables}")

# Recreate papers table if missing
if 'papers' not in tables:
    print("  Recreating papers table...")
    cursor.execute("""
        CREATE TABLE papers (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            journal TEXT,
            abstract TEXT,
            keywords TEXT,
            primary_domain TEXT,
            domains TEXT,
            study_type TEXT,
            source_path TEXT,
            indexed_at TEXT
        )
    """)
    conn.commit()
    print("  ✅ papers table created")

# Check if ingestion_ledger has data we can use
try:
    ledger_count = cursor.execute("SELECT count(*) FROM ingestion_ledger").fetchone()[0]
    print(f"  Ingestion ledger has {ledger_count} entries")
except:
    ledger_count = 0

# Check papers count
papers_count = cursor.execute("SELECT count(*) FROM papers").fetchone()[0]
print(f"  Papers in main table: {papers_count}")

# Check FTS
try:
    fts_count = cursor.execute("SELECT count(*) FROM papers_fts").fetchone()[0]
    print(f"  Papers in FTS: {fts_count}")
except Exception as e:
    print(f"  FTS check failed: {e}")
    fts_count = 0

# Drop and recreate FTS with triggers
print("\nRecreating FTS5 and triggers...")
cursor.execute("DROP TABLE IF EXISTS papers_fts")
cursor.execute("DROP TRIGGER IF EXISTS papers_ai")
cursor.execute("DROP TRIGGER IF EXISTS papers_ad")
cursor.execute("DROP TRIGGER IF EXISTS papers_au")

cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
        doc_id, title, abstract, keywords, authors,
        content='papers', content_rowid='rowid'
    )
""")

cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
        INSERT INTO papers_fts(rowid, doc_id, title, abstract, keywords, authors)
        VALUES (new.rowid, new.doc_id, new.title, new.abstract, new.keywords, new.authors);
    END
""")

cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, doc_id, title, abstract, keywords, authors)
        VALUES ('delete', old.rowid, old.doc_id, old.title, old.abstract, old.keywords, old.authors);
    END
""")

cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
        INSERT INTO papers_fts(papers_fts, rowid, doc_id, title, abstract, keywords, authors)
        VALUES ('delete', old.rowid, old.doc_id, old.title, old.abstract, old.keywords, old.authors);
        INSERT INTO papers_fts(rowid, doc_id, title, abstract, keywords, authors)
        VALUES (new.rowid, new.doc_id, new.title, new.abstract, new.keywords, new.authors);
    END
""")

conn.commit()
print("  ✅ FTS5 and triggers recreated")

if papers_count > 0:
    # Rebuild FTS from existing papers data
    cursor.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    conn.commit()
    new_fts = cursor.execute("SELECT count(*) FROM papers_fts").fetchone()[0]
    print(f"  ✅ FTS rebuilt with {new_fts} entries")
else:
    print("  ⚠️  Papers table is empty - need to re-run ingestion to populate it")

conn.close()
print("\nDone! Re-run ingestion to repopulate the papers table.")
