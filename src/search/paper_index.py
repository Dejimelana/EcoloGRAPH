"""
Paper Index with SQLite FTS5 for keyword search and filtering.

Provides:
- Full-text search with BM25 ranking
- Metadata filters (year, journal, authors, domain)
- Fast pre-filtering for semantic search
"""
import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IndexedPaper:
    """Paper metadata stored in the index."""
    doc_id: str
    title: str
    authors: list[str]
    year: int | None
    journal: str | None
    abstract: str | None
    keywords: list[str]
    primary_domain: str | None
    domains: dict[str, float]  # domain -> score
    study_type: str | None
    source_path: str | None
    indexed_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "primary_domain": self.primary_domain,
            "domains": self.domains,
            "study_type": self.study_type,
            "source_path": self.source_path,
            "indexed_at": self.indexed_at.isoformat()
        }


@dataclass 
class SearchFilters:
    """Filters for paper search."""
    year_min: int | None = None
    year_max: int | None = None
    journals: list[str] | None = None
    authors: list[str] | None = None
    domains: list[str] | None = None
    study_types: list[str] | None = None
    
    def is_empty(self) -> bool:
        return all([
            self.year_min is None,
            self.year_max is None,
            self.journals is None,
            self.authors is None,
            self.domains is None,
            self.study_types is None
        ])


@dataclass
class SearchResult:
    """A search result with ranking score."""
    doc_id: str
    title: str
    score: float  # BM25 or combined score
    year: int | None
    primary_domain: str | None
    snippet: str | None = None


class PaperIndex:
    """
    SQLite-based paper index with FTS5 full-text search.
    
    Features:
    - BM25 keyword ranking
    - Metadata filtering
    - Fast lookups for semantic search pre-filtering
    """
    
    def __init__(self, db_path: str | Path = "data/paper_index.db"):
        """
        Initialize paper index.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"PaperIndex initialized at {self.db_path}")
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Main papers table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,  -- JSON array
                    year INTEGER,
                    journal TEXT,
                    abstract TEXT,
                    keywords TEXT,  -- JSON array
                    primary_domain TEXT,
                    domains TEXT,  -- JSON dict
                    study_type TEXT,
                    source_path TEXT,
                    indexed_at TEXT
                )
            """)
            
            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
                    doc_id,
                    title,
                    abstract,
                    keywords,
                    authors,
                    content='papers',
                    content_rowid='rowid'
                )
            """)
            
            # Triggers to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
                    INSERT INTO papers_fts(rowid, doc_id, title, abstract, keywords, authors)
                    VALUES (new.rowid, new.doc_id, new.title, new.abstract, new.keywords, new.authors);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, doc_id, title, abstract, keywords, authors)
                    VALUES ('delete', old.rowid, old.doc_id, old.title, old.abstract, old.keywords, old.authors);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
                    INSERT INTO papers_fts(papers_fts, rowid, doc_id, title, abstract, keywords, authors)
                    VALUES ('delete', old.rowid, old.doc_id, old.title, old.abstract, old.keywords, old.authors);
                    INSERT INTO papers_fts(rowid, doc_id, title, abstract, keywords, authors)
                    VALUES (new.rowid, new.doc_id, new.title, new.abstract, new.keywords, new.authors);
                END
            """)
            
            # Indexes for filtering
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON papers(primary_domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal ON papers(journal)")
            
            conn.commit()
    
    # --------------------------------------------------------
    # Indexing
    # --------------------------------------------------------
    
    def add_paper(self, paper: IndexedPaper):
        """Add or update a paper in the index."""
        import json
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO papers 
                (doc_id, title, authors, year, journal, abstract, keywords,
                 primary_domain, domains, study_type, source_path, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper.doc_id,
                paper.title,
                json.dumps(paper.authors),
                paper.year,
                paper.journal,
                paper.abstract,
                json.dumps(paper.keywords),
                paper.primary_domain,
                json.dumps(paper.domains),
                paper.study_type,
                paper.source_path,
                paper.indexed_at.isoformat()
            ))
            conn.commit()
        
        logger.debug(f"Indexed paper: {paper.doc_id}")
    
    def add_papers(self, papers: list[IndexedPaper]):
        """Add multiple papers."""
        for paper in papers:
            self.add_paper(paper)
        logger.info(f"Indexed {len(papers)} papers")
    
    def remove_paper(self, doc_id: str):
        """Remove a paper from the index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM papers WHERE doc_id = ?", (doc_id,))
            conn.commit()
    
    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------
    
    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
        offset: int = 0
    ) -> list[SearchResult]:
        """
        Search papers with BM25 ranking and filters.
        
        Args:
            query: Search query (keywords)
            filters: Optional metadata filters
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of SearchResult sorted by relevance
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build query
            if query.strip():
                # FTS5 search with BM25
                sql = """
                    SELECT 
                        p.doc_id,
                        p.title,
                        p.year,
                        p.primary_domain,
                        p.abstract,
                        bm25(papers_fts) as score
                    FROM papers_fts
                    JOIN papers p ON papers_fts.doc_id = p.doc_id
                    WHERE papers_fts MATCH ?
                """
                params = [self._prepare_fts_query(query)]
            else:
                # No query, just filter
                sql = """
                    SELECT 
                        p.doc_id,
                        p.title,
                        p.year,
                        p.primary_domain,
                        p.abstract,
                        0 as score
                    FROM papers p
                    WHERE 1=1
                """
                params = []
            
            # Add filters
            if filters:
                if filters.year_min:
                    sql += " AND p.year >= ?"
                    params.append(filters.year_min)
                
                if filters.year_max:
                    sql += " AND p.year <= ?"
                    params.append(filters.year_max)
                
                if filters.journals:
                    placeholders = ",".join("?" * len(filters.journals))
                    sql += f" AND p.journal IN ({placeholders})"
                    params.extend(filters.journals)
                
                if filters.domains:
                    placeholders = ",".join("?" * len(filters.domains))
                    sql += f" AND p.primary_domain IN ({placeholders})"
                    params.extend(filters.domains)
                
                if filters.study_types:
                    placeholders = ",".join("?" * len(filters.study_types))
                    sql += f" AND p.study_type IN ({placeholders})"
                    params.extend(filters.study_types)
                
                if filters.authors:
                    # JSON array search
                    author_conditions = " OR ".join(
                        "p.authors LIKE ?" for _ in filters.authors
                    )
                    sql += f" AND ({author_conditions})"
                    params.extend(f"%{a}%" for a in filters.authors)
            
            # Order and limit
            sql += " ORDER BY score LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(sql, params)
            
            results = []
            for row in cursor:
                # Create snippet from abstract
                abstract = row["abstract"] or ""
                snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                
                results.append(SearchResult(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    score=abs(row["score"]),  # BM25 returns negative
                    year=row["year"],
                    primary_domain=row["primary_domain"],
                    snippet=snippet
                ))
            
            return results
    
    def get_doc_ids_by_filter(
        self,
        filters: SearchFilters,
        limit: int = 100
    ) -> list[str]:
        """Get doc_ids matching filters (for Qdrant pre-filtering)."""
        results = self.search("", filters=filters, limit=limit)
        return [r.doc_id for r in results]
    
    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query for FTS5 (handle special chars, broad matching)."""
        import re
        # Remove special chars that break FTS5
        clean = re.sub(r'[^\w\s]', ' ', query)
        tokens = [t for t in clean.split() if len(t) >= 2]
        if not tokens:
            return query
        # Use prefix matching with OR for broader results
        return " OR ".join(f"{t}*" for t in tokens)
    
    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------
    
    def get_paper(self, doc_id: str) -> IndexedPaper | None:
        """Get a paper by doc_id."""
        import json
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM papers WHERE doc_id = ?", 
                (doc_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return IndexedPaper(
                doc_id=row["doc_id"],
                title=row["title"],
                authors=json.loads(row["authors"] or "[]"),
                year=row["year"],
                journal=row["journal"],
                abstract=row["abstract"],
                keywords=json.loads(row["keywords"] or "[]"),
                primary_domain=row["primary_domain"],
                domains=json.loads(row["domains"] or "{}"),
                study_type=row["study_type"],
                source_path=row["source_path"],
                indexed_at=datetime.fromisoformat(row["indexed_at"])
            )
    
    def count(self) -> int:
        """Get total number of indexed papers."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM papers")
            return cursor.fetchone()[0]
    
    def get_domains(self) -> list[tuple[str, int]]:
        """Get domain distribution."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT primary_domain, COUNT(*) as cnt 
                FROM papers 
                WHERE primary_domain IS NOT NULL
                GROUP BY primary_domain
                ORDER BY cnt DESC
            """)
            return [(row[0], row[1]) for row in cursor]

    def get_all_papers(self, limit: int = 200, offset: int = 0) -> list[IndexedPaper]:
        """Get all indexed papers, ordered by year (newest first)."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM papers ORDER BY year DESC, title ASC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            papers = []
            for row in cursor:
                papers.append(IndexedPaper(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    authors=json.loads(row["authors"] or "[]"),
                    year=row["year"],
                    journal=row["journal"],
                    abstract=row["abstract"],
                    keywords=json.loads(row["keywords"] or "[]"),
                    primary_domain=row["primary_domain"],
                    domains=json.loads(row["domains"] or "{}"),
                    study_type=row["study_type"],
                    source_path=row["source_path"],
                    indexed_at=datetime.fromisoformat(row["indexed_at"]),
                ))
            return papers
    
    def get_year_range(self) -> tuple[int, int]:
        """Get min/max years."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT MIN(year), MAX(year) FROM papers WHERE year IS NOT NULL
            """)
            row = cursor.fetchone()
            return (row[0] or 0, row[1] or 0)
    
    def clear(self):
        """Clear all indexed papers."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM papers")
            conn.execute("DELETE FROM papers_fts")
            conn.commit()
        logger.warning("Paper index cleared")
