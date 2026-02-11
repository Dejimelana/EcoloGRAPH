"""
Qdrant Vector Store for semantic search over ecological data.

Provides:
- Document chunk embeddings
- Semantic similarity search
- Hybrid retrieval (semantic + keyword)
- Metadata filtering
"""
import logging
from dataclasses import dataclass, field
from typing import Any
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SearchParams,
    HnswConfigDiff,
    OptimizersConfigDiff
)
from sentence_transformers import SentenceTransformer

from ..ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


# Default embedding model - good balance of quality and speed
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions
VECTOR_SIZE = 384


@dataclass
class SearchResult:
    """A single search result with metadata."""
    
    chunk_id: str
    doc_id: str
    text: str
    score: float
    
    # Metadata
    domain: str | None = None
    section: str | None = None
    page: int | None = None
    
    # Extracted entities (if available)
    species: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "domain": self.domain,
            "section": self.section,
            "page": self.page,
            "species": self.species,
            "locations": self.locations
        }


class VectorStore:
    """
    Qdrant-based vector store for semantic search.
    
    Supports:
    - Adding document chunks with embeddings
    - Semantic similarity search
    - Filtered search by domain, species, etc.
    - Hybrid retrieval combining semantic and keyword matching
    """
    
    def __init__(
        self,
        collection_name: str = "ecograph_chunks",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = DEFAULT_MODEL,
        vector_size: int = VECTOR_SIZE
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            embedding_model: Sentence transformer model name
            vector_size: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        else:
            logger.info(f"Using existing collection: {self.collection_name}")
    
    # --------------------------------------------------------
    # Indexing
    # --------------------------------------------------------
    
    def add_chunk(
        self,
        chunk: DocumentChunk,
        domain: str | None = None,
        species: list[str] | None = None,
        locations: list[str] | None = None
    ):
        """
        Add a single chunk to the vector store.
        
        Args:
            chunk: Document chunk to index
            domain: Scientific domain of the chunk
            species: Species mentioned in the chunk
            locations: Locations mentioned in the chunk
        """
        # Generate embedding
        embedding = self.embedder.encode(chunk.text)
        # Convert to list if numpy array
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Generate point ID from chunk_id
        point_id = self._generate_point_id(chunk.chunk_id)
        
        # Build payload
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "section": chunk.section,
            "page": chunk.page,
            "domain": domain,
            "species": species or [],
            "locations": locations or []
        }
        
        # Upsert point
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
    
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        domain: str | None = None,
        batch_size: int = 200,  # Increased from 100 for 2x speedup
        doc_metadata: dict | None = None  # NEW: metadata with title, source_path
    ):
        """
        Add multiple chunks to the vector store with optimized batching.
        
        Args:
            chunks: List of document chunks
            domain: Domain for all chunks (or None)
            batch_size: Number of chunks per batch (default: 200)
            doc_metadata: Optional dict with 'title' and 'source_path' for searchability
        """
        import time
        
        total_chunks = len(chunks)
        logger.info(f"Indexing {total_chunks} chunks in batches of {batch_size}...")
        
        start_time = time.time()
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        for i in range(0, total_chunks, batch_size):
            batch_start = time.time()
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Generate embeddings for batch
            texts = [c.text for c in batch]
            embeddings = self.embedder.encode(
                texts,
                show_progress_bar=False,  # Avoid duplicate progress bars
                convert_to_numpy=False
            )
            # Convert to list if numpy array
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            # Build points
            points = []
            for chunk, embedding in zip(batch, embeddings):
                point_id = self._generate_point_id(chunk.chunk_id)
                
                metadata = doc_metadata or {}
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": metadata.get('title', 'Unknown'),
                    "source_path": metadata.get('source_path', 'Unknown'),
                    "text": chunk.text,
                    "section": chunk.section,
                    "page": chunk.page,
                    "char_count": len(chunk.text),
                    "domain": domain
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=False  # Don't wait for acknowledgment (faster)
            )
            
            batch_time = time.time() - batch_start
            logger.info(
                f"Batch {batch_num}/{total_batches} ({len(batch)} chunks) "
                f"indexed in {batch_time:.2f}s"
            )
        
        total_time = time.time() - start_time
        chunks_per_sec = total_chunks / total_time if total_time > 0 else 0
        
        logger.info(
            f"✅ Indexed {total_chunks} chunks in {total_time:.2f}s "
            f"({chunks_per_sec:.1f} chunks/sec)"
        )
    
    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------
    
    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
        domain: str | None = None,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        species: list[str] | None = None
    ) -> list[SearchResult]:
        """
        Semantic search for relevant chunks.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            domain: Filter by domain (optional)
            doc_id: Filter by document ID (optional)
            species: Filter by species mentioned (optional)
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        # Build filter
        filter_conditions = []
        
        if domain:
            filter_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        
        if doc_id:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )
        
        if doc_ids:
            filter_conditions.append(
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            )
        
        if species:
            filter_conditions.append(
                FieldCondition(key="species", match=MatchAny(any=species))
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            search_params=SearchParams(hnsw_ef=128)
        )
        
        # Convert to SearchResult
        return [
            SearchResult(
                chunk_id=r.payload.get("chunk_id", ""),
                doc_id=r.payload.get("doc_id", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                domain=r.payload.get("domain"),
                section=r.payload.get("section"),
                page=r.payload.get("page_start"),
                species=r.payload.get("species", []),
                locations=r.payload.get("locations", [])
            )
            for r in results
        ]
    
    def search_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
        exclude_same_doc: bool = True
    ) -> list[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum results
            exclude_same_doc: Exclude chunks from same document
            
        Returns:
            List of similar chunks
        """
        point_id = self._generate_point_id(chunk_id)
        
        # Get the chunk's vector
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True
        )
        
        if not points:
            return []
        
        point = points[0]
        doc_id = point.payload.get("doc_id")
        
        # Build filter to exclude same document
        query_filter = None
        if exclude_same_doc and doc_id:
            query_filter = Filter(
                must_not=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            )
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=point.vector,
            query_filter=query_filter,
            limit=limit
        )
        
        return [
            SearchResult(
                chunk_id=r.payload.get("chunk_id", ""),
                doc_id=r.payload.get("doc_id", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                domain=r.payload.get("domain"),
                section=r.payload.get("section"),
                page=r.payload.get("page_start")
            )
            for r in results
        ]
    
    def search_by_species(
        self,
        species_name: str,
        query: str | None = None,
        limit: int = 20
    ) -> list[SearchResult]:
        """
        Find all chunks mentioning a species, optionally with semantic ranking.
        
        Args:
            species_name: Species to search for
            query: Optional semantic query for ranking
            limit: Maximum results
            
        Returns:
            Chunks mentioning the species
        """
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="species", 
                    match=MatchAny(any=[species_name])
                )
            ]
        )
        
        if query:
            # Semantic search with species filter
            query_embedding = self.embedder.encode(query)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_condition,
                limit=limit
            )
        else:
            # Just scroll through all matching chunks
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=limit,
                with_payload=True
            )
        
        return [
            SearchResult(
                chunk_id=r.payload.get("chunk_id", ""),
                doc_id=r.payload.get("doc_id", ""),
                text=r.payload.get("text", ""),
                score=getattr(r, 'score', 1.0),
                domain=r.payload.get("domain"),
                section=r.payload.get("section"),
                page=r.payload.get("page_start"),
                species=r.payload.get("species", [])
            )
            for r in results
        ]
    
    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------
    
    def hybrid_search(
        self,
        query: str,
        keywords: list[str] | None = None,
        limit: int = 10,
        semantic_weight: float = 0.7
    ) -> list[SearchResult]:
        """
        Hybrid search combining semantic similarity and keyword matching.
        
        Args:
            query: Search query
            keywords: Additional keywords to boost
            limit: Maximum results
            semantic_weight: Weight for semantic score (0-1)
            
        Returns:
            Combined and reranked results
        """
        # Get semantic results
        semantic_results = self.search(query, limit=limit * 2)
        
        # If no keywords, just return semantic results
        if not keywords:
            return semantic_results[:limit]
        
        # Score adjustment based on keyword presence
        keyword_weight = 1 - semantic_weight
        
        scored_results = []
        for result in semantic_results:
            text_lower = result.text.lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            
            # Combined score
            combined_score = (
                semantic_weight * result.score +
                keyword_weight * keyword_score
            )
            
            result.score = combined_score
            scored_results.append(result)
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        return scored_results[:limit]
    
    # --------------------------------------------------------
    # Management
    # --------------------------------------------------------
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "collection": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.name
        }
    
    def delete_document(self, doc_id: str):
        """Delete all chunks for a document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            )
        )
        logger.info(f"Deleted chunks for document: {doc_id}")
    
    def clear(self):
        """Clear all vectors from the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        logger.warning("Vector store cleared!")
    
    def _generate_point_id(self, chunk_id: str) -> int:
        """Generate numeric point ID from chunk ID string."""
        # Use hash to get consistent numeric ID
        return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)
    
    def close(self):
        """Close the client connection."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
