"""
Repair Neo4j Paper nodes -- sync titles, years, abstracts from SQLite PaperIndex.

Uses DIRECT sqlite3 + neo4j-driver to avoid heavy import chains.

Usage:
    python scripts/repair_neo4j_titles.py
"""
import sqlite3
import sys
import io
from pathlib import Path
from neo4j import GraphDatabase

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -- Config --
SQLITE_DB = Path(__file__).parent.parent / "data" / "paper_index.db"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"


def load_papers_from_sqlite(db_path: Path) -> list[dict]:
    """Read all papers from SQLite PaperIndex."""
    if not db_path.exists():
        print(f"[ERROR] SQLite DB not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    col_info = conn.execute("PRAGMA table_info(papers)").fetchall()
    col_names = {row[1] for row in col_info}

    select_cols = ["doc_id", "title", "year", "abstract", "authors", "keywords", "journal"]
    if "doi" in col_names:
        select_cols.append("doi")
    if "source_path" in col_names:
        select_cols.append("source_path")

    cursor = conn.execute(f"SELECT {', '.join(select_cols)} FROM papers")
    papers = []
    for row in cursor:
        paper = {col: row[col] for col in select_cols}
        if "doi" not in paper:
            paper["doi"] = None
        papers.append(paper)
    conn.close()
    return papers


def main():
    papers = load_papers_from_sqlite(SQLITE_DB)
    print(f"[INFO] Found {len(papers)} papers in SQLite ({SQLITE_DB})")

    if not papers:
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"[OK] Connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Neo4j: {e}")
        return

    updated = 0
    skipped = 0

    with driver.session(database=NEO4J_DATABASE) as session:
        for paper in papers:
            doc_id = paper["doc_id"]
            title = paper["title"]
            year = paper["year"]
            abstract = paper["abstract"]
            doi = paper.get("doi")
            journal = paper.get("journal")

            if not title or title == "Unknown":
                skipped += 1
                continue

            query = """
            MERGE (p:Paper {doc_id: $doc_id})
            SET p.title = $title,
                p.year = $year,
                p.abstract = $abstract,
                p.doi = $doi,
                p.journal = $journal
            RETURN p.title AS t
            """

            try:
                result = session.run(query, {
                    "doc_id": doc_id,
                    "title": title,
                    "year": year,
                    "abstract": abstract,
                    "doi": doi,
                    "journal": journal,
                })
                record = result.single()
                if record:
                    updated += 1
                    safe_title = title[:60].encode("ascii", "replace").decode()
                    print(f"  [OK] {doc_id[:12]} -> {safe_title}")
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [ERR] {doc_id[:12]} error: {e}")
                skipped += 1

    driver.close()
    print(f"\n[DONE] Repair complete: {updated} updated, {skipped} skipped")

    # Verify
    try:
        driver2 = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver2.session(database=NEO4J_DATABASE) as s:
            res = s.run("MATCH (p:Paper) WHERE p.title <> 'Unknown' RETURN count(p) AS c")
            good = res.single()["c"]
            res2 = s.run("MATCH (p:Paper) WHERE p.title = 'Unknown' OR p.title IS NULL RETURN count(p) AS c")
            bad = res2.single()["c"]
            print(f"[VERIFY] {good} papers with titles, {bad} still 'Unknown'")
        driver2.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
