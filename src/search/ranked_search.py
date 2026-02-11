"""
Ranked Search combining SQLite keyword search with Qdrant semantic reranking.

Provides:
- Two-stage search: keyword filtering → semantic reranking
- Combined scoring from BM25 and embeddings
- Efficient pre-filtering for large document sets
"""
import logging
import time
from dataclasses import dataclass
from typing import Any

from .paper_index import PaperIndex, SearchFilters, SearchResult as KeywordResult
from .query_logger import QueryLogger
from ..retrieval.vector_store import VectorStore, SearchResult as SemanticResult

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """Final ranked search result with combined scores."""
    doc_id: str
    title: str
    
    # Scores
    keyword_score: float
    semantic_score: float
    combined_score: float
    
    # Metadata
    year: int | None = None
    primary_domain: str | None = None
    snippet: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "keyword_score": self.keyword_score,
            "semantic_score": self.semantic_score,
            "combined_score": self.combined_score,
            "year": self.year,
            "primary_domain": self.primary_domain,
            "snippet": self.snippet
        }


class RankedSearch:
    """
    Two-stage ranked search combining keyword and semantic approaches.
    
    Strategy:
    1. SQLite FTS5 for fast keyword matching + metadata filtering
    2. Qdrant for semantic reranking of top candidates
    3. Combined scoring with configurable weights
    """
    
    def __init__(
        self,
        paper_index: PaperIndex | None = None,
        vector_store: VectorStore | None = None,
        query_logger: QueryLogger | None = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Initialize ranked search.
        
        Args:
            paper_index: SQLite paper index
            vector_store: Qdrant vector store
            query_logger: Query logger for tracking
            keyword_weight: Weight for keyword score (0-1)
            semantic_weight: Weight for semantic score (0-1)
        """
        self.paper_index = paper_index
        self.vector_store = vector_store
        self.query_logger = query_logger
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        
        # Normalize weights
        total = keyword_weight + semantic_weight
        if total > 0:
            self.keyword_weight = keyword_weight / total
            self.semantic_weight = semantic_weight / total
        
        logger.info(
            f"RankedSearch initialized "
            f"(keyword={self.keyword_weight:.0%}, semantic={self.semantic_weight:.0%})"
        )
    
    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 20,
        candidate_multiplier: int = 5
    ) -> list[RankedResult]:
        """
        Perform ranked search with filtering.
        
        Args:
            query: Search query
            filters: Metadata filters
            limit: Final number of results
            candidate_multiplier: How many candidates to fetch for reranking
            
        Returns:
            List of RankedResult sorted by combined score
        """
        start_time = time.time()
        
        # Stage 1: Keyword search with filters
        keyword_results = []
        if self.paper_index:
            keyword_results = self.paper_index.search(
                query=query,
                filters=filters,
                limit=limit * candidate_multiplier
            )
        
        if not keyword_results:
            # No keyword results, try semantic only
            if self.vector_store:
                semantic_only = self._semantic_search_with_filters(
                    query, filters, limit
                )
                self._log_query(query, filters, len(semantic_only), start_time)
                return semantic_only
            return []
        
        # Stage 2: Semantic reranking
        if self.vector_store and query.strip():
            results = self._rerank_with_semantic(
                query, keyword_results, limit
            )
        else:
            # No vector store, use keyword results only
            results = [
                RankedResult(
                    doc_id=r.doc_id,
                    title=r.title,
                    keyword_score=r.score,
                    semantic_score=0.0,
                    combined_score=r.score,
                    year=r.year,
                    primary_domain=r.primary_domain,
                    snippet=r.snippet
                )
                for r in keyword_results[:limit]
            ]
        
        # Log query
        self._log_query(query, filters, len(results), start_time)
        
        return results
    
    def _rerank_with_semantic(
        self,
        query: str,
        keyword_results: list[KeywordResult],
        limit: int
    ) -> list[RankedResult]:
        """Rerank keyword results using semantic similarity.
        
        Optimized: uses server-side doc_ids filter in Qdrant instead of
        fetching all vectors and filtering in Python.
        """
        # Get doc_ids from keyword results
        doc_ids = list({r.doc_id for r in keyword_results})
        
        # Create lookup for keyword scores
        keyword_scores = {r.doc_id: r for r in keyword_results}
        
        # Normalize keyword scores to 0-1
        max_kw_score = max(r.score for r in keyword_results) if keyword_results else 1
        if max_kw_score == 0:
            max_kw_score = 1
        
        # Semantic search WITH doc_id filter (server-side in Qdrant)
        # This is the key optimization: only scans candidate chunks, not the
        # entire collection
        try:
            semantic_results = self.vector_store.search(
                query=query,
                limit=limit * 3,  # Get enough for good reranking
                score_threshold=0.0,
                doc_ids=doc_ids,  # Server-side filter!
            )
            
            # Aggregate chunk scores per doc_id (best chunk wins)
            semantic_scores: dict[str, float] = {}
            for r in semantic_results:
                if r.doc_id not in semantic_scores or r.score > semantic_scores[r.doc_id]:
                    semantic_scores[r.doc_id] = r.score
                    
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            semantic_scores = {}
        
        # Combine scores
        combined_results = []
        for doc_id in doc_ids:
            kw_result = keyword_scores[doc_id]
            
            # Normalized keyword score
            norm_kw_score = kw_result.score / max_kw_score
            
            # Semantic score (0 if not found)
            sem_score = semantic_scores.get(doc_id, 0.0)
            
            # Combined score
            combined = (
                self.keyword_weight * norm_kw_score +
                self.semantic_weight * sem_score
            )
            
            combined_results.append(RankedResult(
                doc_id=doc_id,
                title=kw_result.title,
                keyword_score=norm_kw_score,
                semantic_score=sem_score,
                combined_score=combined,
                year=kw_result.year,
                primary_domain=kw_result.primary_domain,
                snippet=kw_result.snippet
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return combined_results[:limit]
    
    def _semantic_search_with_filters(
        self,
        query: str,
        filters: SearchFilters | None,
        limit: int
    ) -> list[RankedResult]:
        """Semantic-only search when no keyword results."""
        try:
            # Get domain filter if set
            domain = None
            if filters and filters.domains and len(filters.domains) == 1:
                domain = filters.domains[0]
            
            results = self.vector_store.search(
                query=query,
                limit=limit,
                domain=domain
            )
            
            return [
                RankedResult(
                    doc_id=r.doc_id,
                    title="",  # Not available from vector search
                    keyword_score=0.0,
                    semantic_score=r.score,
                    combined_score=r.score,
                    primary_domain=r.domain,
                    snippet=r.text[:200] if r.text else None
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _log_query(
        self,
        query: str,
        filters: SearchFilters | None,
        result_count: int,
        start_time: float
    ):
        """Log query to query logger."""
        if not self.query_logger:
            return
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        filter_dict = {}
        if filters:
            if filters.year_min:
                filter_dict["year_min"] = filters.year_min
            if filters.year_max:
                filter_dict["year_max"] = filters.year_max
            if filters.domains:
                filter_dict["domains"] = filters.domains
            if filters.journals:
                filter_dict["journals"] = filters.journals
        
        self.query_logger.log_query(
            query=query,
            filters=filter_dict,
            result_count=result_count,
            execution_time_ms=execution_time
        )
    
    # --------------------------------------------------------
    # Convenience Methods
    # --------------------------------------------------------
    
    def search_recent_years(
        self,
        query: str,
        years: int = 5,
        limit: int = 20
    ) -> list[RankedResult]:
        """Search papers from the last N years."""
        from datetime import datetime
        current_year = datetime.now().year
        
        filters = SearchFilters(year_min=current_year - years)
        return self.search(query, filters=filters, limit=limit)
    
    def search_by_domain(
        self,
        query: str,
        domain: str,
        limit: int = 20
    ) -> list[RankedResult]:
        """Search within a specific domain."""
        filters = SearchFilters(domains=[domain])
        return self.search(query, filters=filters, limit=limit)
    
    def search_by_author(
        self,
        query: str,
        author: str,
        limit: int = 20
    ) -> list[RankedResult]:
        """Search papers by a specific author."""
        filters = SearchFilters(authors=[author])
        return self.search(query, filters=filters, limit=limit)
    
    def get_top_papers(
        self,
        query: str,
        limit: int = 20,
        years: int | None = None,
        domain: str | None = None
    ) -> list[RankedResult]:
        """
        Get top papers for a query with optional filters.
        
        This is the main entry point for most searches.
        """
        filters = SearchFilters()
        
        if years:
            from datetime import datetime
            filters.year_min = datetime.now().year - years
        
        if domain:
            filters.domains = [domain]
        
        return self.search(query, filters=filters, limit=limit)
