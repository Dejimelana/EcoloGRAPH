"""
Query Logger for tracking search history and database updates.

Provides:
- Query history with timestamps
- Database update logging (add/delete/modify)
- Analytics on search patterns
"""
import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import json

logger = logging.getLogger(__name__)


@dataclass
class QueryLogEntry:
    """A logged query."""
    query_id: int
    query: str
    filters: dict
    result_count: int
    execution_time_ms: float
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "filters": self.filters,
            "result_count": self.result_count,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class UpdateLogEntry:
    """A logged database update."""
    update_id: int
    action: str  # add, delete, modify
    doc_id: str
    details: dict
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "update_id": self.update_id,
            "action": self.action,
            "doc_id": self.doc_id,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class QueryLogger:
    """
    SQLite-based logging for queries and database updates.
    
    Features:
    - Query history tracking
    - Update/modification logging
    - Search analytics
    """
    
    def __init__(self, db_path: str | Path = "data/query_log.db"):
        """
        Initialize query logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"QueryLogger initialized at {self.db_path}")
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Query log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    filters TEXT,  -- JSON
                    result_count INTEGER,
                    execution_time_ms REAL,
                    timestamp TEXT
                )
            """)
            
            # Update log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS update_log (
                    update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    doc_id TEXT,
                    details TEXT,  -- JSON
                    timestamp TEXT
                )
            """)
            
            # Indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_time ON query_log(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_update_time ON update_log(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_update_doc ON update_log(doc_id)"
            )
            
            conn.commit()
    
    # --------------------------------------------------------
    # Query Logging
    # --------------------------------------------------------
    
    def log_query(
        self,
        query: str,
        filters: dict | None = None,
        result_count: int = 0,
        execution_time_ms: float = 0.0
    ) -> int:
        """
        Log a search query.
        
        Returns:
            query_id of the logged entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO query_log (query, filters, result_count, execution_time_ms, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                query,
                json.dumps(filters or {}),
                result_count,
                execution_time_ms,
                datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_queries(self, limit: int = 20) -> list[QueryLogEntry]:
        """Get recent queries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM query_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [
                QueryLogEntry(
                    query_id=row["query_id"],
                    query=row["query"],
                    filters=json.loads(row["filters"] or "{}"),
                    result_count=row["result_count"],
                    execution_time_ms=row["execution_time_ms"],
                    timestamp=datetime.fromisoformat(row["timestamp"])
                )
                for row in cursor
            ]
    
    def get_popular_queries(self, days: int = 30, limit: int = 10) -> list[tuple[str, int]]:
        """Get most popular queries in the last N days."""
        cutoff = datetime.now().isoformat()[:10]  # Simplified
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT query, COUNT(*) as cnt 
                FROM query_log
                WHERE query != ''
                GROUP BY query
                ORDER BY cnt DESC
                LIMIT ?
            """, (limit,))
            
            return [(row[0], row[1]) for row in cursor]
    
    # --------------------------------------------------------
    # Update Logging
    # --------------------------------------------------------
    
    def log_update(
        self,
        action: str,
        doc_id: str,
        details: dict | None = None
    ) -> int:
        """
        Log a database update.
        
        Args:
            action: Type of update (add, delete, modify)
            doc_id: Document ID affected
            details: Additional details
            
        Returns:
            update_id of the logged entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO update_log (action, doc_id, details, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                action,
                doc_id,
                json.dumps(details or {}),
                datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def log_paper_added(self, doc_id: str, title: str = ""):
        """Convenience method for logging paper addition."""
        self.log_update("add", doc_id, {"title": title})
    
    def log_paper_deleted(self, doc_id: str):
        """Convenience method for logging paper deletion."""
        self.log_update("delete", doc_id)
    
    def log_paper_modified(self, doc_id: str, fields: list[str]):
        """Convenience method for logging paper modification."""
        self.log_update("modify", doc_id, {"fields_changed": fields})
    
    def get_recent_updates(self, limit: int = 50) -> list[UpdateLogEntry]:
        """Get recent database updates."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM update_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            return [
                UpdateLogEntry(
                    update_id=row["update_id"],
                    action=row["action"],
                    doc_id=row["doc_id"],
                    details=json.loads(row["details"] or "{}"),
                    timestamp=datetime.fromisoformat(row["timestamp"])
                )
                for row in cursor
            ]
    
    def get_paper_history(self, doc_id: str) -> list[UpdateLogEntry]:
        """Get all updates for a specific paper."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM update_log 
                WHERE doc_id = ?
                ORDER BY timestamp DESC
            """, (doc_id,))
            
            return [
                UpdateLogEntry(
                    update_id=row["update_id"],
                    action=row["action"],
                    doc_id=row["doc_id"],
                    details=json.loads(row["details"] or "{}"),
                    timestamp=datetime.fromisoformat(row["timestamp"])
                )
                for row in cursor
            ]
    
    # --------------------------------------------------------
    # Analytics
    # --------------------------------------------------------
    
    def get_stats(self) -> dict:
        """Get logging statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Query stats
            q_count = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
            q_today = conn.execute("""
                SELECT COUNT(*) FROM query_log 
                WHERE timestamp >= date('now')
            """).fetchone()[0]
            
            # Update stats
            u_count = conn.execute("SELECT COUNT(*) FROM update_log").fetchone()[0]
            
            add_count = conn.execute(
                "SELECT COUNT(*) FROM update_log WHERE action = 'add'"
            ).fetchone()[0]
            
            return {
                "total_queries": q_count,
                "queries_today": q_today,
                "total_updates": u_count,
                "papers_added": add_count,
                "avg_query_time_ms": self._get_avg_query_time()
            }
    
    def _get_avg_query_time(self) -> float:
        """Get average query execution time."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT AVG(execution_time_ms) FROM query_log"
            ).fetchone()[0]
            return result or 0.0
    
    def clear_logs(self, before_date: datetime | None = None):
        """Clear logs, optionally before a specific date."""
        with sqlite3.connect(self.db_path) as conn:
            if before_date:
                date_str = before_date.isoformat()
                conn.execute(
                    "DELETE FROM query_log WHERE timestamp < ?", 
                    (date_str,)
                )
                conn.execute(
                    "DELETE FROM update_log WHERE timestamp < ?", 
                    (date_str,)
                )
            else:
                conn.execute("DELETE FROM query_log")
                conn.execute("DELETE FROM update_log")
            conn.commit()
        logger.warning("Logs cleared")
