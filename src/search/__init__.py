"""
Search module - Keyword ranking, filtering, and query logging.

Provides:
- PaperIndex: SQLite FTS5 for BM25 keyword search
- QueryLogger: Search history and update tracking
- RankedSearch: Hybrid keyword + semantic search
"""
from .paper_index import PaperIndex, IndexedPaper, SearchFilters, SearchResult
from .query_logger import QueryLogger, QueryLogEntry, UpdateLogEntry
from .ranked_search import RankedSearch, RankedResult

__all__ = [
    "PaperIndex",
    "IndexedPaper",
    "SearchFilters",
    "SearchResult",
    "QueryLogger",
    "QueryLogEntry",
    "UpdateLogEntry",
    "RankedSearch",
    "RankedResult"
]
