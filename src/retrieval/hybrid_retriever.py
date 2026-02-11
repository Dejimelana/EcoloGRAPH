"""
Hybrid Retriever combining Graph + Vector search.

Provides:
- Multi-source retrieval (Neo4j + Qdrant)
- Context assembly for RAG
- Relevance-based deduplication
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from .vector_store import VectorStore, SearchResult
from ..graph.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Combined context from multiple sources for RAG."""
    
    query: str
    
    # Vector search results
    semantic_chunks: list[SearchResult] = field(default_factory=list)
    
    # Graph query results
    related_species: list[dict] = field(default_factory=list)
    species_measurements: list[dict] = field(default_factory=list)
    ecological_relations: list[dict] = field(default_factory=list)
    
    # Assembled context
    context_text: str = ""
    
    # Metadata
    sources_used: list[str] = field(default_factory=list)
    total_chunks: int = 0
    
    def to_prompt_context(self) -> str:
        """Format context for LLM prompt."""
        sections = []
        
        if self.semantic_chunks:
            sections.append("## Relevant Text Passages\n")
            for i, chunk in enumerate(self.semantic_chunks[:5], 1):
                sections.append(
                    f"### Passage {i} (from {chunk.doc_id}, score: {chunk.score:.2f})\n"
                    f"{chunk.text[:800]}...\n"
                )
        
        if self.species_measurements:
            sections.append("\n## Species Measurements\n")
            for m in self.species_measurements[:10]:
                sections.append(
                    f"- {m.get('parameter')}: {m.get('value')} {m.get('unit')} "
                    f"(source: {m.get('source_paper', 'unknown')})\n"
                )
        
        if self.ecological_relations:
            sections.append("\n## Ecological Relationships\n")
            for r in self.ecological_relations[:10]:
                sections.append(
                    f"- {r.get('from_species')} → {r.get('relation_type')} → {r.get('to_species')}\n"
                )
        
        return "".join(sections)


class HybridRetriever:
    """
    Combines vector search and graph queries for comprehensive retrieval.
    
    Strategy:
    1. Semantic search for relevant text chunks
    2. Extract entities (species, locations) from query
    3. Graph queries for structured data about entities
    4. Combine and deduplicate results
    """
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        graph_builder: GraphBuilder | None = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            graph_builder: Neo4j graph builder instance
        """
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        
        logger.info(
            f"HybridRetriever initialized "
            f"(vectors={'yes' if vector_store else 'no'}, "
            f"graph={'yes' if graph_builder else 'no'})"
        )
    
    def retrieve(
        self,
        query: str,
        species_names: list[str] | None = None,
        domain: str | None = None,
        max_chunks: int = 10,
        include_graph: bool = True
    ) -> RetrievalContext:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            species_names: Known species to include (optional)
            domain: Filter by scientific domain (optional)
            max_chunks: Maximum text chunks to retrieve
            include_graph: Whether to query the graph
            
        Returns:
            RetrievalContext with all relevant information
        """
        context = RetrievalContext(query=query)
        
        # 1. Semantic search
        if self.vector_store:
            try:
                chunks = self.vector_store.search(
                    query=query,
                    limit=max_chunks,
                    domain=domain
                )
                context.semantic_chunks = chunks
                context.sources_used.append("vector_store")
                context.total_chunks = len(chunks)
                
                # Extract species from chunks if not provided
                if not species_names:
                    species_names = self._extract_species_from_chunks(chunks)
                    
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # 2. Graph queries
        if include_graph and self.graph_builder and species_names:
            try:
                for species in species_names[:3]:  # Limit to avoid too many queries
                    # Get measurements
                    measurements = self.graph_builder.get_species_measurements(species)
                    context.species_measurements.extend(measurements)
                    
                    # Get ecological network
                    relations = self.graph_builder.get_ecological_network(species, depth=1)
                    context.ecological_relations.extend(relations)
                
                context.sources_used.append("knowledge_graph")
                
            except Exception as e:
                logger.warning(f"Graph queries failed: {e}")
        
        # 3. Assemble context
        context.context_text = context.to_prompt_context()
        
        return context
    
    def retrieve_for_species(
        self,
        species_name: str,
        query: str | None = None,
        include_related: bool = True
    ) -> RetrievalContext:
        """
        Retrieve all information about a specific species.
        
        Args:
            species_name: Species to retrieve data for
            query: Optional query to rank chunks
            include_related: Include related species
            
        Returns:
            RetrievalContext focused on the species
        """
        context = RetrievalContext(query=query or f"Information about {species_name}")
        
        # Vector search for chunks mentioning species
        if self.vector_store:
            try:
                chunks = self.vector_store.search_by_species(
                    species_name=species_name,
                    query=query,
                    limit=15
                )
                context.semantic_chunks = chunks
                context.sources_used.append("vector_store")
                
            except Exception as e:
                logger.warning(f"Species vector search failed: {e}")
        
        # Graph queries
        if self.graph_builder:
            try:
                # Get measurements
                measurements = self.graph_builder.get_species_measurements(species_name)
                context.species_measurements = measurements
                
                # Get ecological network
                if include_related:
                    relations = self.graph_builder.get_ecological_network(
                        species_name, 
                        depth=2
                    )
                    context.ecological_relations = relations
                
                context.sources_used.append("knowledge_graph")
                
            except Exception as e:
                logger.warning(f"Species graph queries failed: {e}")
        
        context.context_text = context.to_prompt_context()
        return context
    
    def find_cross_document_evidence(
        self,
        claim: str,
        min_sources: int = 2
    ) -> list[SearchResult]:
        """
        Find evidence for a claim across multiple documents.
        
        Args:
            claim: Statement to find evidence for
            min_sources: Minimum number of different documents
            
        Returns:
            Chunks from different documents supporting the claim
        """
        if not self.vector_store:
            return []
        
        # Get more results than needed to find diverse sources
        results = self.vector_store.search(
            query=claim,
            limit=20,
            score_threshold=0.4
        )
        
        # Group by document and select best from each
        by_doc: dict[str, SearchResult] = {}
        for result in results:
            doc_id = result.doc_id
            if doc_id not in by_doc or result.score > by_doc[doc_id].score:
                by_doc[doc_id] = result
        
        # Return if we have enough sources
        evidence = list(by_doc.values())
        evidence.sort(key=lambda x: x.score, reverse=True)
        
        if len(evidence) >= min_sources:
            return evidence
        
        return evidence  # Return what we have
    
    def _extract_species_from_chunks(self, chunks: list[SearchResult]) -> list[str]:
        """Extract species names from chunk metadata."""
        species = set()
        for chunk in chunks:
            if chunk.species:
                species.update(chunk.species)
        return list(species)[:5]  # Limit
