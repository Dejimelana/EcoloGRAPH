"""
Ingestion ledger tracking module.

Tracks which papers have been successfully indexed in each store
(SQLite, Qdrant, Neo4j) for improved traceability and debugging.
"""
import sqlite3
from datetime import datetime
from pathlib import Path


class IngestionLedger:
    """Track ingestion status across multiple stores."""
    
    STORES = ["sqlite", "qdrant", "neo4j"]
    STATUSES = ["pending", "success", "failed"]
    
    def __init__(self, db_path: str | Path):
        """
        Initialize ledger with database path.
        
        Args:
            db_path: Path to SQLite database (typically paper_index.db)
        """
        self.db_path = Path(db_path)
        self._init_ledger()
    
    def _init_ledger(self):
        """Create ledger table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_ledger (
                    doc_id TEXT NOT NULL,
                    store TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'success', 'failed')),
                    timestamp TEXT NOT NULL,
                    error_message TEXT,
                    PRIMARY KEY (doc_id, store)
                )
            """)
            conn.commit()
    
    def record(self, doc_id: str, store: str, status: str, error: str | None = None):
        """
        Record ingestion attempt for a paper in a specific store.
        
        Args:
            doc_id: Document ID
            store: Store name ('sqlite', 'qdrant', 'neo4j')
            status: Status ('pending', 'success', 'failed')
            error: Optional error message if failed
        """
        if store not in self.STORES:
            raise ValueError(f"Unknown store: {store}. Must be one of {self.STORES}")
        if status not in self.STATUSES:
            raise ValueError(f"Unknown status: {status}. Must be one of {self.STATUSES}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_ledger 
                (doc_id, store, status, timestamp, error_message)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, store, status, datetime.now().isoformat(), error))
            conn.commit()
    
    def get_status(self, doc_id: str) -> dict[str, str]:
        """
        Get ingestion status for a document across all stores.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary mapping store name to status
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT store, status, timestamp, error_message
                FROM ingestion_ledger
                WHERE doc_id = ?
            """, (doc_id,)).fetchall()
        
        return {
            row[0]: {
                "status": row[1],
                "timestamp": row[2],
                "error": row[3]
            }
            for row in result
        }
    
    def get_partial_ingestions(self) -> list[tuple[str, dict]]:
        """
        Find papers that are not indexed in all stores.
        
        Returns:
            List of (doc_id, status_dict) tuples for partially indexed papers
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find docs that don't have success status in all 3 stores
            result = conn.execute("""
                SELECT doc_id
                FROM ingestion_ledger
                GROUP BY doc_id
                HAVING COUNT(CASE WHEN status = 'success' THEN 1 END) < 3
                ORDER BY MIN(timestamp) DESC
            """).fetchall()
        
        partial = []
        for (doc_id,) in result:
            status = self.get_status(doc_id)
            partial.append((doc_id, status))
        
        return partial
    
    def get_failed_ingestions(self, store: str | None = None) -> list[tuple[str, str, str]]:
        """
        Get all failed ingestion attempts.
        
        Args:
            store: Optional store filter
            
        Returns:
            List of (doc_id, store, error_message) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            if store:
                result = conn.execute("""
                    SELECT doc_id, store, error_message
                    FROM ingestion_ledger
                    WHERE status = 'failed' AND store = ?
                    ORDER BY timestamp DESC
                """, (store,)).fetchall()
            else:
                result = conn.execute("""
                    SELECT doc_id, store, error_message
                    FROM ingestion_ledger
                    WHERE status = 'failed'
                    ORDER BY timestamp DESC
                """).fetchall()
        
        return result
    
    def get_stats(self) -> dict:
        """
        Get aggregate statistics.
        
        Returns:
            Dictionary with counts by store and status
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT store, status, COUNT(*) as count
                FROM ingestion_ledger
                GROUP BY store, status
                ORDER BY store, status
            """).fetchall()
        
        stats = {store: {} for store in self.STORES}
        for store, status, count in result:
            stats[store][status] = count
        
        # Add computed metrics
        total_docs = len(set(row[0] for row in conn.execute("SELECT doc_id FROM ingestion_ledger").fetchall()))
        fully_indexed = conn.execute("""
            SELECT COUNT(DISTINCT doc_id)
            FROM ingestion_ledger
            WHERE doc_id IN (
                SELECT doc_id
                FROM ingestion_ledger
                WHERE status = 'success'
                GROUP BY doc_id
                HAVING COUNT(DISTINCT store) = 3
            )
        """).fetchone()[0]
        
        stats["summary"] = {
            "total_docs": total_docs,
            "fully_indexed": fully_indexed,
            "partially_indexed": total_docs - fully_indexed
        }
        
        return stats
