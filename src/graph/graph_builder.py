"""
Neo4j Graph Builder for EcoloGRAPH.

Builds a knowledge graph from extracted ecological data with full traceability.

Node Types:
- Paper: Scientific publication
- Species: Taxonomic entity
- Location: Geographic location
- Measurement: Quantitative data
- Trait: Species trait/characteristic
- Author: Paper author

Relationship Types:
- MENTIONS: Paper mentions Species
- MEASURED_AT: Measurement at Location
- HAS_TRAIT: Species has Trait
- RELATES_TO: Ecological relationship between Species
- AUTHORED_BY: Paper authored by Author
- CITES: Paper cites Paper
"""
import logging
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

from neo4j import GraphDatabase, Driver

from ..core.schemas import (
    ExtractionResult, 
    SpeciesMention, 
    Measurement, 
    Location,
    EcologicalRelation,
    SpeciesTrait,
    Citation  # NEW: for citation network
)

# Query result caching
from diskcache import Cache
from functools import wraps
import hashlib
import json

logger = logging.getLogger(__name__)

# Module-level cache for GraphBuilder queries
_query_cache = Cache(".cache/neo4j_queries", size_limit=100 * 1024 * 1024)  # 100 MB


def cached_query(ttl=300):
    """
    Decorator for caching Neo4j query results.
    
    Args:
        ttl: Time-to-live in seconds (default: 5 minutes)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            result = _query_cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit: {func.__name__}({args})")
                return result
            
            # Execute query
            result = func(self, *args, **kwargs)
            
            # Store in cache
            _query_cache.set(cache_key, result, expire=ttl)
            logger.debug(f"Cache miss: {func.__name__}({args}) - stored with TTL={ttl}s")
            
            return result
        return wrapper
    return decorator


# ============================================================
# Schema Definition
# ============================================================

SCHEMA_CONSTRAINTS = """
// Paper node uniqueness
CREATE CONSTRAINT paper_id IF NOT EXISTS
FOR (p:Paper) REQUIRE p.doc_id IS UNIQUE;

// Species node uniqueness
CREATE CONSTRAINT species_name IF NOT EXISTS
FOR (s:Species) REQUIRE s.scientific_name IS UNIQUE;

// Location node uniqueness (composite)
CREATE CONSTRAINT location_id IF NOT EXISTS
FOR (l:Location) REQUIRE l.location_id IS UNIQUE;

// Measurement node uniqueness
CREATE CONSTRAINT measurement_id IF NOT EXISTS
FOR (m:Measurement) REQUIRE m.measurement_id IS UNIQUE;

// Author node uniqueness
CREATE CONSTRAINT author_id IF NOT EXISTS
FOR (a:Author) REQUIRE a.author_id IS UNIQUE;
"""

SCHEMA_INDEXES = """
// Full-text search on species
CREATE FULLTEXT INDEX species_search IF NOT EXISTS
FOR (s:Species) ON EACH [s.scientific_name, s.common_names];

// Full-text search on papers
CREATE FULLTEXT INDEX paper_search IF NOT EXISTS
FOR (p:Paper) ON EACH [p.title, p.abstract];

// Species taxonomy index
CREATE INDEX species_taxonomy IF NOT EXISTS
FOR (s:Species) ON (s.family, s.genus);

// Location geographic index
CREATE INDEX location_geo IF NOT EXISTS
FOR (l:Location) ON (l.latitude, l.longitude);

// Measurement parameter index
CREATE INDEX measurement_param IF NOT EXISTS
FOR (m:Measurement) ON (m.parameter);

// Paper year index
CREATE INDEX paper_year IF NOT EXISTS
FOR (p:Paper) ON (p.year);
"""


@dataclass
class GraphStats:
    """Statistics about the graph."""
    paper_count: int = 0
    species_count: int = 0
    location_count: int = 0
    measurement_count: int = 0
    relationship_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "papers": self.paper_count,
            "species": self.species_count,
            "locations": self.location_count,
            "measurements": self.measurement_count,
            "relationships": self.relationship_count
        }


class GraphBuilder:
    """
    Builds and manages the Neo4j knowledge graph.
    
    Provides methods to:
    - Initialize schema (constraints, indexes)
    - Add papers with extracted data
    - Query the graph
    - Get statistics
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize graph builder.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self.uri = uri
        self.database = database
        
        try:
            self._driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self._driver:
            self._driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # --------------------------------------------------------
    # Schema Management
    # --------------------------------------------------------
    
    def initialize_schema(self):
        """Create constraints and indexes."""
        with self._driver.session(database=self.database) as session:
            # Create constraints
            for statement in SCHEMA_CONSTRAINTS.strip().split(";"):
                stmt = statement.strip()
                if stmt and not stmt.startswith("//"):
                    try:
                        session.run(stmt)
                    except Exception as e:
                        logger.debug(f"Constraint may already exist: {e}")
            
            # Create indexes
            for statement in SCHEMA_INDEXES.strip().split(";"):
                stmt = statement.strip()
                if stmt and not stmt.startswith("//"):
                    try:
                        session.run(stmt)
                    except Exception as e:
                        logger.debug(f"Index may already exist: {e}")
        
        logger.info("Graph schema initialized")
    
    # --------------------------------------------------------
    # Paper Management
    # --------------------------------------------------------
    
    def add_paper(
        self,
        doc_id: str,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        doi: str | None = None,
        abstract: str | None = None,
        source_path: str | None = None,
        metadata: dict | None = None
    ) -> str:
        """
        Add a paper node to the graph.
        
        Returns:
            The doc_id of the created/updated paper
        """
        query = """
        MERGE (p:Paper {doc_id: $doc_id})
        SET p.title = $title,
            p.year = $year,
            p.doi = $doi,
            p.abstract = $abstract,
            p.source_path = $source_path,
            p.updated_at = datetime()
        RETURN p.doc_id as doc_id
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {
                "doc_id": doc_id,
                "title": title,
                "year": year,
                "doi": doi,
                "abstract": abstract,
                "source_path": source_path
            })
            
            record = result.single()
            
            # Add authors
            if authors:
                for author in authors:
                    try:
                        self._add_author_relationship(session, doc_id, author)
                    except Exception as e:
                        logger.warning(f"Error indexing author {author}: {e}")
        
        logger.debug(f"Indexed {len(authors) if authors else 0} authors for {doc_id}")
        return doc_id
    
    def add_citations(
        self,
        doc_id: str,
        citations: list[Citation]
    ):
        """
        Add citations from a paper to the knowledge graph.
        
        Creates Citation nodes and CITES relationships.
        Optionally creates CITES_PAPER relationships if citation matches existing paper.
        
        Args:
            doc_id: Source paper doc_id
            citations: List of Citation objects
        """
        if not citations:
            return
        
        with self._driver.session(database=self.database) as session:
            for citation in citations:
                try:
                    # Create Citation node
                    query = """
                    MERGE (c:Citation {
                        title: $title,
                        year: $year
                    })
                    ON CREATE SET
                        c.authors = $authors,
                        c.journal = $journal,
                        c.doi = $doi,
                        c.first_seen = datetime()
                    
                    // Link to citing paper
                    WITH c
                    MATCH (p:Paper {doc_id: $doc_id})
                    MERGE (p)-[:CITES]->(c)
                    
                    RETURN c
                    """
                    
                    session.run(query, {
                        "title": citation.title,
                        "year": citation.year,
                        "authors": citation.authors,
                        "journal": citation.journal,
                        "doi": citation.doi,
                        "doc_id": doc_id
                    })
                    
                    # If citation matches an existing paper, create direct link
                    if citation.matches_doc_id and citation.match_confidence > 0.7:
                        match_query = """
                        MATCH (citing:Paper {doc_id: $citing_doc_id})
                        MATCH (cited:Paper {doc_id: $cited_doc_id})
                        MERGE (citing)-[r:CITES_PAPER {
                            confidence: $confidence
                        }]->(cited)
                        """
                        
                        session.run(match_query, {
                            "citing_doc_id": doc_id,
                            "cited_doc_id": citation.matches_doc_id,
                            "confidence": citation.match_confidence
                        })
                
                except Exception as e:
                    logger.warning(f"Error adding citation '{citation.title[:50]}...': {e}")
        
        logger.debug(f"Added {len(citations)} citations for {doc_id}")
    
    def _add_author_relationship(self, session, doc_id: str, author_name: str):
        """Add author node and relationship to paper."""
        query = """
        MATCH (p:Paper {doc_id: $doc_id})
        MERGE (a:Author {author_id: $author_id})
        SET a.name = $name
        MERGE (p)-[:AUTHORED_BY]->(a)
        """
        
        # Simple author ID from name
        author_id = author_name.lower().replace(" ", "_").replace(".", "")
        
        session.run(query, {
            "doc_id": doc_id,
            "author_id": author_id,
            "name": author_name
        })
    
    # --------------------------------------------------------
    # Species Management
    # --------------------------------------------------------
    
    def add_species_mention(
        self,
        doc_id: str,
        species: SpeciesMention,
        chunk_id: str | None = None
    ):
        """Add a species mention from a paper."""
        query = """
        MATCH (p:Paper {doc_id: $doc_id})
        MERGE (s:Species {scientific_name: $scientific_name})
        SET s.common_names = COALESCE($common_names, s.common_names),
            s.kingdom = COALESCE($kingdom, s.kingdom),
            s.phylum = COALESCE($phylum, s.phylum),
            s.class = COALESCE($class, s.class),
            s.order = COALESCE($order, s.order),
            s.family = COALESCE($family, s.family),
            s.genus = COALESCE($genus, s.genus),
            s.aphia_id = COALESCE($aphia_id, s.aphia_id),
            s.gbif_key = COALESCE($gbif_key, s.gbif_key)
        MERGE (p)-[r:MENTIONS]->(s)
        SET r.original_name = $original_name,
            r.chunk_id = $chunk_id,
            r.confidence = $confidence
        """
        
        with self._driver.session(database=self.database) as session:
            session.run(query, {
                "doc_id": doc_id,
                "scientific_name": species.scientific_name or species.original_name,
                "original_name": species.original_name,
                "common_names": species.common_names,
                "kingdom": species.kingdom,
                "phylum": species.phylum,
                "class": species.class_name,
                "order": species.order,
                "family": species.family,
                "genus": species.genus,
                "aphia_id": species.aphia_id,
                "gbif_key": species.gbif_key,
                "chunk_id": chunk_id,
                "confidence": species.source.confidence if species.source else 1.0
            })
    
    # --------------------------------------------------------
    # Location Management
    # --------------------------------------------------------
    
    def add_location(
        self,
        doc_id: str,
        location: Location,
        chunk_id: str | None = None
    ) -> str:
        """Add a location from a paper."""
        # Generate location ID
        location_id = f"{location.name}_{location.region or 'unknown'}".lower().replace(" ", "_")
        
        query = """
        MATCH (p:Paper {doc_id: $doc_id})
        MERGE (l:Location {location_id: $location_id})
        SET l.name = $name,
            l.region = $region,
            l.country = $country,
            l.habitat_type = $habitat_type,
            l.latitude = $latitude,
            l.longitude = $longitude
        MERGE (p)-[r:REFERENCES_LOCATION]->(l)
        SET r.chunk_id = $chunk_id
        RETURN l.location_id as location_id
        """
        
        coords = location.coordinates
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {
                "doc_id": doc_id,
                "location_id": location_id,
                "name": location.name,
                "region": location.region,
                "country": location.country,
                "habitat_type": location.habitat_type,
                "latitude": coords.latitude if coords else None,
                "longitude": coords.longitude if coords else None,
                "chunk_id": chunk_id
            })
            
            return result.single()["location_id"]
    
    # --------------------------------------------------------
    # Measurement Management
    # --------------------------------------------------------
    
    def add_measurement(
        self,
        doc_id: str,
        measurement: Measurement,
        chunk_id: str | None = None
    ):
        """Add a measurement from a paper."""
        import hashlib
        
        # Generate measurement ID
        m_str = f"{doc_id}_{measurement.parameter}_{measurement.value}_{measurement.species or ''}"
        measurement_id = hashlib.md5(m_str.encode()).hexdigest()[:12]
        
        query = """
        MATCH (p:Paper {doc_id: $doc_id})
        MERGE (m:Measurement {measurement_id: $measurement_id})
        SET m.parameter = $parameter,
            m.value = $value,
            m.unit = $unit,
            m.min_value = $min_value,
            m.max_value = $max_value,
            m.std_error = $std_error,
            m.sample_size = $sample_size,
            m.life_stage = $life_stage
        MERGE (p)-[r:CONTAINS]->(m)
        SET r.chunk_id = $chunk_id
        
        // Link to species if specified
        WITH m
        OPTIONAL MATCH (s:Species {scientific_name: $species})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:HAS_MEASUREMENT]->(m)
        )
        """
        
        with self._driver.session(database=self.database) as session:
            # Handle unit as either enum or string
            unit_value = None
            if measurement.unit:
                unit_value = measurement.unit.value if hasattr(measurement.unit, 'value') else str(measurement.unit)
            
            session.run(query, {
                "doc_id": doc_id,
                "measurement_id": measurement_id,
                "parameter": measurement.parameter,
                "value": measurement.value,
                "unit": unit_value,
                "min_value": measurement.min_value,
                "max_value": measurement.max_value,
                "std_error": measurement.std_error,
                "sample_size": measurement.sample_size,
                "life_stage": measurement.life_stage,
                "species": measurement.species,
                "chunk_id": chunk_id
            })
    
    # --------------------------------------------------------
    # Ecological Relationships
    # --------------------------------------------------------
    
    def add_ecological_relation(
        self,
        doc_id: str,
        relation: EcologicalRelation,
        chunk_id: str | None = None
    ):
        """Add an ecological relationship between species."""
        query = """
        MATCH (p:Paper {doc_id: $doc_id})
        MERGE (s1:Species {scientific_name: $subject})
        MERGE (s2:Species {scientific_name: $object})
        MERGE (s1)-[r:RELATES_TO {type: $relation_type}]->(s2)
        SET r.description = $description,
            r.source_paper = $doc_id,
            r.chunk_id = $chunk_id
        """
        
        with self._driver.session(database=self.database) as session:
            # Handle relation_type as either enum or string
            relation_type_value = relation.relation_type.value if hasattr(relation.relation_type, 'value') else str(relation.relation_type)
            
            session.run(query, {
                "doc_id": doc_id,
                "subject": relation.subject,
                "object": relation.object,
                "relation_type": relation_type_value,
                "description": relation.description,
                "chunk_id": chunk_id
            })
    
    # --------------------------------------------------------
    # Bulk Import
    # --------------------------------------------------------
    
    def add_extraction_result(
        self,
        doc_id: str,
        result: ExtractionResult,
        paper_metadata: dict | None = None
    ):
        """
        Add complete extraction result to graph.
        
        This is the main entry point for ingesting extracted data.
        """
        # Add paper node
        self.add_paper(
            doc_id=doc_id,
            title=paper_metadata.get("title", "Unknown") if paper_metadata else "Unknown",
            authors=paper_metadata.get("authors") if paper_metadata else None,
            year=paper_metadata.get("year") if paper_metadata else None,
            doi=paper_metadata.get("doi") if paper_metadata else None,
            abstract=paper_metadata.get("abstract") if paper_metadata else None
        )
        
        # Add species mentions
        for species in result.species:
            chunk_id = species.source.chunk_id if species.source else None
            self.add_species_mention(doc_id, species, chunk_id)
        
        # Add locations
        for location in result.locations:
            chunk_id = location.source.chunk_id if location.source else None
            self.add_location(doc_id, location, chunk_id)
        
        # Add measurements
        for measurement in result.measurements:
            chunk_id = measurement.source.chunk_id if measurement.source else None
            self.add_measurement(doc_id, measurement, chunk_id)
        
        # Add ecological relations
        for relation in result.relations:
            chunk_id = relation.source.chunk_id if relation.source else None
            self.add_ecological_relation(doc_id, relation, chunk_id)
        
        logger.info(
            f"Added extraction result for {doc_id}: "
            f"{len(result.species)} species, {len(result.measurements)} measurements"
        )
    
    # --------------------------------------------------------
    # Queries
    # --------------------------------------------------------
    
    @cached_query(ttl=600)  # 10 minutes cache (species data more stable)
    def get_species_papers(self, scientific_name: str) -> list[dict]:
        """Get all papers mentioning a species."""
        query = """
        MATCH (p:Paper)-[r:MENTIONS]->(s:Species {scientific_name: $name})
        RETURN p.doc_id as doc_id, 
               p.title as title, 
               p.year as year,
               r.confidence as confidence
        ORDER BY p.year DESC
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {"name": scientific_name})
            return [dict(record) for record in result]
    
    def get_species_measurements(self, scientific_name: str) -> list[dict]:
        """Get all measurements for a species."""
        query = """
        MATCH (s:Species {scientific_name: $name})-[:HAS_MEASUREMENT]->(m:Measurement)
        OPTIONAL MATCH (p:Paper)-[:CONTAINS]->(m)
        RETURN m.parameter as parameter,
               m.value as value,
               m.unit as unit,
               m.life_stage as life_stage,
               p.doc_id as source_paper,
               p.year as year
        ORDER BY m.parameter
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {"name": scientific_name})
            return [dict(record) for record in result]
    
    def get_ecological_network(self, scientific_name: str, depth: int = 2) -> list[dict]:
        """Get ecological relationships network around a species."""
        query = """
        MATCH path = (s:Species {scientific_name: $name})-[:RELATES_TO*1..$depth]-(other:Species)
        UNWIND relationships(path) as rel
        WITH DISTINCT startNode(rel) as from, endNode(rel) as to, rel
        RETURN from.scientific_name as from_species,
               to.scientific_name as to_species,
               rel.type as relation_type,
               rel.description as description
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {"name": scientific_name, "depth": depth})
            return [dict(record) for record in result]
    
    @cached_query(ttl=300)  # 5 minutes cache
    def get_paper_metadata(self, doc_id: str) -> dict | None:
        """
        Get full metadata for a paper node.
        
        Returns:
            Dictionary with paper metadata or None if not found
        """
        query = """
        MATCH (p:Paper {doc_id: $doc_id})
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:MENTIONS]->(s:Species)
        OPTIONAL MATCH (p)-[:REFERENCES_LOCATION]->(l:Location)
        WITH p, 
             collect(DISTINCT a.name) as authors,
             collect(DISTINCT s.scientific_name) as species,
             collect(DISTINCT l.name) as locations
        RETURN p.doc_id as doc_id,
               p.title as title,
               p.abstract as abstract,
               p.year as year,
               p.doi as doi,
               p.source_path as source_path,
               authors,
               species,
               locations
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {"doc_id": doc_id})
            record = result.single()
            return dict(record) if record else None
    
    def get_paper_chunks(self, doc_id: str) -> list[dict]:
        """
        Get all chunks for a paper from Qdrant vector store.
        
        Args:
            doc_id: Document ID to fetch chunks for
            
        Returns:
            List of chunk dictionaries with text, section, page, etc.
        """
        try:
            from src.retrieval.vector_store import VectorStore
            
            vs = VectorStore()
            
            # Use scroll API to get ALL chunks for this doc_id
            # (search with empty query + filter is more efficient than scroll for single doc)
            chunks_result, _ = vs.client.scroll(
                collection_name=vs.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id}
                        }
                    ]
                },
                limit=1000,  # Max chunks per paper
                with_payload=True,
                with_vectors=False  # Don't need vectors
            )
            
            # Format results
            chunks = []
            for point in chunks_result:
                payload = point.payload
                chunks.append({
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text", ""),
                    "section": payload.get("section"),
                    "page": payload.get("page"),
                    "word_count": len(payload.get("text", "").split()),
                    "char_count": len(payload.get("text", "")),
                    "chunk_idx": chunks.index(point) if hasattr(point, 'index') else None
                })
            
            # Sort by chunk_idx if available (preserve document order)
            chunks.sort(key=lambda x: x.get("chunk_idx") or 0)
            
            logger.info(f"Retrieved {len(chunks)} chunks for doc_id={doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to fetch chunks for {doc_id}: {e}")
            return []
    
    def get_node_connections(
        self, 
        node_id: str, 
        node_type: str,
        max_depth: int = 1
    ) -> dict:
        """
        Get ego network (connections) for any node.
        
        Args:
            node_id: ID of the node (doc_id for Paper, scientific_name for Species, etc.)
            node_type: Type of node ('Paper', 'Species', 'Location')
            max_depth: Maximum relationship depth to explore
            
        Returns:
            Dictionary with nodes and relationships
        """
        # Build query based on node type
        if node_type == "Paper":
            id_field = "doc_id"
        elif node_type == "Species":
            id_field = "scientific_name"
        elif node_type == "Location":
            id_field = "location_id"
        else:
            logger.error(f"Unknown node type: {node_type}")
            return {"nodes": [], "relationships": []}
        
        query = f"""
        MATCH (n:{node_type} {{{id_field}: $node_id}})
        OPTIONAL MATCH path = (n)-[r*1..{max_depth}]-(connected)
        WITH n, 
             collect(DISTINCT connected) as connected_nodes,
             collect(DISTINCT relationships(path)) as paths
        UNWIND connected_nodes as conn
        WITH n, conn, paths,
             labels(conn)[0] as conn_type
        RETURN n as center_node,
               collect(DISTINCT {{
                   id: CASE 
                       WHEN 'Paper' IN labels(conn) THEN conn.doc_id
                       WHEN 'Species' IN labels(conn) THEN conn.scientific_name
                       WHEN 'Location' IN labels(conn) THEN conn.location_id
                       ELSE id(conn)
                   END,
                   type: conn_type,
                   properties: properties(conn)
               }}) as connected_nodes,
               paths
        """
        
        with self._driver.session(database=self.database) as session:
            result = session.run(query, {"node_id": node_id})
            record = result.single()
            
            if not record:
                return {"nodes": [], "relationships": []}
            
            return {
                "center": dict(record["center_node"]),
                "nodes": record["connected_nodes"],
                "paths": record["paths"]
            }
    
    def get_stats(self) -> GraphStats:
        """Get graph statistics."""
        queries = {
            "papers": "MATCH (p:Paper) RETURN count(p) as count",
            "species": "MATCH (s:Species) RETURN count(s) as count",
            "locations": "MATCH (l:Location) RETURN count(l) as count",
            "measurements": "MATCH (m:Measurement) RETURN count(m) as count",
            "relationships": "MATCH ()-[r]->() RETURN count(r) as count"
        }
        
        stats = GraphStats()
        
        with self._driver.session(database=self.database) as session:
            for key, query in queries.items():
                result = session.run(query)
                count = result.single()["count"]
                setattr(stats, f"{key[:-1] if key.endswith('s') else key}_count", count)
        
        return stats
    
    def clear_graph(self):
        """Clear all nodes and relationships (use with caution!)."""
        query = "MATCH (n) DETACH DELETE n"
        
        with self._driver.session(database=self.database) as session:
            session.run(query)
        
        logger.warning("Graph cleared!")
