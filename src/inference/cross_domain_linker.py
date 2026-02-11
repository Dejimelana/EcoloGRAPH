"""
Cross-Domain Linker for discovering connections between scientific domains.

Identifies patterns where:
- Same species studied in different domains
- Same methodology applied across domains
- Shared measurements with different interpretations
- Geographic co-occurrence across studies
"""
import logging
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from ..core.domain_registry import DomainType, DomainRegistry
from ..graph.graph_builder import GraphBuilder
from ..retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class LinkType(str, Enum):
    """Types of cross-domain links."""
    SHARED_SPECIES = "shared_species"  # Same species in different domains
    SHARED_LOCATION = "shared_location"  # Same location studied
    SHARED_METHODOLOGY = "shared_methodology"  # Same method in different domains
    COMPLEMENTARY_DATA = "complementary_data"  # Different measurements of same entity
    ECOLOGICAL_CASCADE = "ecological_cascade"  # Effects propagating across domains
    TAXONOMIC_BRIDGE = "taxonomic_bridge"  # Related taxa in different domains


@dataclass
class CrossDomainLink:
    """A discovered link between two domains."""
    
    # Required fields (no defaults)
    link_id: str
    link_type: LinkType
    source_domain: DomainType
    source_entity: str  # Species name, location, etc.
    target_domain: DomainType
    target_entity: str
    
    # Optional fields (with defaults)
    source_papers: list[str] = field(default_factory=list)
    target_papers: list[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0
    shared_attributes: dict = field(default_factory=dict)
    evidence_chunks: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "link_id": self.link_id,
            "link_type": self.link_type.value,
            "source_domain": self.source_domain.value,
            "source_entity": self.source_entity,
            "source_papers": self.source_papers,
            "target_domain": self.target_domain.value,
            "target_entity": self.target_entity,
            "target_papers": self.target_papers,
            "description": self.description,
            "confidence": self.confidence,
            "shared_attributes": self.shared_attributes
        }


class CrossDomainLinker:
    """
    Discovers and scores cross-domain links in the knowledge graph.
    
    Strategies:
    1. Entity co-occurrence: Same entity in papers from different domains
    2. Semantic similarity: Chunks with high similarity across domains
    3. Graph traversal: Paths connecting nodes from different domains
    4. Taxonomic proximity: Related species across domains
    """
    
    def __init__(
        self,
        graph_builder: GraphBuilder | None = None,
        vector_store: VectorStore | None = None
    ):
        """
        Initialize cross-domain linker.
        
        Args:
            graph_builder: Neo4j graph instance
            vector_store: Qdrant vector store instance
        """
        self.graph = graph_builder
        self.vectors = vector_store
        
        # Domain compatibility matrix (which domains can meaningfully connect)
        self.domain_affinity = self._build_affinity_matrix()
        
        logger.info("CrossDomainLinker initialized")
    
    def _build_affinity_matrix(self) -> dict[tuple[DomainType, DomainType], float]:
        """Build matrix of domain pair affinities (0-1)."""
        affinities = {}
        
        # High affinity pairs
        high_affinity = [
            (DomainType.MARINE_ECOLOGY, DomainType.FRESHWATER_ECOLOGY),
            (DomainType.MARINE_ECOLOGY, DomainType.CONSERVATION),
            (DomainType.GENETICS, DomainType.CONSERVATION),
            (DomainType.GENETICS, DomainType.POPULATION_MODELING),
            (DomainType.BOTANY, DomainType.ENTOMOLOGY),  # Pollination
            (DomainType.TOXICOLOGY, DomainType.CONSERVATION),
            (DomainType.MICROBIOLOGY, DomainType.SOIL_SCIENCE),
            (DomainType.REMOTE_SENSING, DomainType.SPATIAL_ECOLOGY),
            (DomainType.HYDROLOGY, DomainType.FRESHWATER_ECOLOGY),
            (DomainType.PARASITOLOGY, DomainType.ZOOLOGY),
            (DomainType.ORNITHOLOGY, DomainType.CONSERVATION),
            (DomainType.ENTOMOLOGY, DomainType.CONSERVATION),
            # AI/Computational
            (DomainType.MACHINE_LEARNING, DomainType.DEEP_LEARNING),
            (DomainType.COMPUTER_VISION, DomainType.DEEP_LEARNING),
            (DomainType.SOUNDSCAPE_ECOLOGY, DomainType.ORNITHOLOGY),
            (DomainType.COMPUTER_VISION, DomainType.REMOTE_SENSING),
            (DomainType.MACHINE_LEARNING, DomainType.BIOINFORMATICS),
            (DomainType.AI_MODELING, DomainType.POPULATION_MODELING),
            # Organismal Biology
            (DomainType.ETHOLOGY, DomainType.ZOOLOGY),
            (DomainType.PHYSIOLOGY, DomainType.ZOOLOGY),
            (DomainType.BIOTIC_INTERACTIONS, DomainType.CONSERVATION),
            (DomainType.BIOGEOGRAPHY, DomainType.CONSERVATION),
            (DomainType.BIOGEOGRAPHY, DomainType.SPATIAL_ECOLOGY),
        ]
        
        for d1, d2 in high_affinity:
            affinities[(d1, d2)] = 0.9
            affinities[(d2, d1)] = 0.9
        
        # Medium affinity pairs
        medium_affinity = [
            (DomainType.MARINE_ECOLOGY, DomainType.MICROBIOLOGY),
            (DomainType.GENETICS, DomainType.TOXICOLOGY),
            (DomainType.PALEOECOLOGY, DomainType.CONSERVATION),
            (DomainType.NETWORK_ECOLOGY, DomainType.MARINE_ECOLOGY),
            # AI cross-domain
            (DomainType.COMPUTER_VISION, DomainType.ENTOMOLOGY),
            (DomainType.SOUNDSCAPE_ECOLOGY, DomainType.MARINE_ECOLOGY),
            (DomainType.MACHINE_LEARNING, DomainType.CONSERVATION),
            (DomainType.AI_MODELING, DomainType.SPATIAL_ECOLOGY),
            (DomainType.DEEP_LEARNING, DomainType.SOUNDSCAPE_ECOLOGY),
            # Organismal / Earth cross-domain
            (DomainType.ETHOLOGY, DomainType.SOUNDSCAPE_ECOLOGY),
            (DomainType.PHYSIOLOGY, DomainType.TOXICOLOGY),
            (DomainType.BIOTIC_INTERACTIONS, DomainType.NETWORK_ECOLOGY),
            (DomainType.GEOLOGY, DomainType.PALEOECOLOGY),
            (DomainType.GEOLOGY, DomainType.HYDROLOGY),
            (DomainType.BIOGEOGRAPHY, DomainType.CLIMATE_SCIENCE),
            (DomainType.ETHOLOGY, DomainType.ORNITHOLOGY),
        ]
        
        for d1, d2 in medium_affinity:
            affinities[(d1, d2)] = 0.6
            affinities[(d2, d1)] = 0.6
        
        return affinities
    
    def get_domain_affinity(self, domain1: DomainType, domain2: DomainType) -> float:
        """Get affinity score between two domains."""
        if domain1 == domain2:
            return 1.0
        return self.domain_affinity.get((domain1, domain2), 0.3)
    
    # --------------------------------------------------------
    # Link Discovery
    # --------------------------------------------------------
    
    def find_shared_species_links(
        self,
        species_name: str,
        min_domains: int = 2
    ) -> list[CrossDomainLink]:
        """
        Find domains that have studied the same species.
        
        Args:
            species_name: Scientific name to search
            min_domains: Minimum domains to be considered a link
            
        Returns:
            Links between domains studying this species
        """
        if not self.graph:
            return []
        
        links = []
        
        # Query: Get all papers mentioning this species with their domains
        query = """
        MATCH (p:Paper)-[:MENTIONS]->(s:Species {scientific_name: $name})
        WHERE p.domain IS NOT NULL
        RETURN p.domain as domain, 
               collect(p.doc_id) as papers,
               collect(p.title) as titles
        """
        
        try:
            with self.graph._driver.session(database=self.graph.database) as session:
                result = session.run(query, {"name": species_name})
                domain_papers = [(r["domain"], r["papers"], r["titles"]) for r in result]
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
        
        if len(domain_papers) < min_domains:
            return []
        
        # Create links between each pair of domains
        import hashlib
        for i, (domain1, papers1, _) in enumerate(domain_papers):
            for domain2, papers2, _ in domain_papers[i+1:]:
                link_id = hashlib.md5(
                    f"{species_name}:{domain1}:{domain2}".encode()
                ).hexdigest()[:12]
                
                try:
                    d1 = DomainType(domain1)
                    d2 = DomainType(domain2)
                except ValueError:
                    continue
                
                affinity = self.get_domain_affinity(d1, d2)
                
                link = CrossDomainLink(
                    link_id=link_id,
                    link_type=LinkType.SHARED_SPECIES,
                    source_domain=d1,
                    source_entity=species_name,
                    source_papers=papers1,
                    target_domain=d2,
                    target_entity=species_name,
                    target_papers=papers2,
                    description=f"{species_name} studied in both {d1.value} and {d2.value}",
                    confidence=affinity,
                    shared_attributes={"species": species_name}
                )
                links.append(link)
        
        return links
    
    def find_semantic_bridges(
        self,
        source_domain: DomainType,
        target_domain: DomainType,
        top_k: int = 5
    ) -> list[CrossDomainLink]:
        """
        Find semantically similar content across two domains.
        
        Uses vector similarity to find chunks from different domains
        that discuss related concepts.
        """
        if not self.vectors:
            return []
        
        links = []
        
        # Get sample chunks from source domain
        try:
            source_results, _ = self.vectors.client.scroll(
                collection_name=self.vectors.collection_name,
                scroll_filter={
                    "must": [{"key": "domain", "match": {"value": source_domain.value}}]
                },
                limit=20,
                with_payload=True,
                with_vectors=True
            )
        except Exception as e:
            logger.error(f"Vector scroll failed: {e}")
            return []
        
        # For each source chunk, find similar in target domain
        import hashlib
        for source_point in source_results[:10]:
            try:
                similar = self.vectors.client.search(
                    collection_name=self.vectors.collection_name,
                    query_vector=source_point.vector,
                    query_filter={
                        "must": [{"key": "domain", "match": {"value": target_domain.value}}]
                    },
                    limit=1,
                    score_threshold=0.7
                )
                
                if similar:
                    match = similar[0]
                    link_id = hashlib.md5(
                        f"{source_point.id}:{match.id}".encode()
                    ).hexdigest()[:12]
                    
                    link = CrossDomainLink(
                        link_id=link_id,
                        link_type=LinkType.COMPLEMENTARY_DATA,
                        source_domain=source_domain,
                        source_entity=source_point.payload.get("chunk_id", ""),
                        source_papers=[source_point.payload.get("doc_id", "")],
                        target_domain=target_domain,
                        target_entity=match.payload.get("chunk_id", ""),
                        target_papers=[match.payload.get("doc_id", "")],
                        description="Semantically similar content across domains",
                        confidence=match.score,
                        evidence_chunks=[
                            source_point.payload.get("text", "")[:200],
                            match.payload.get("text", "")[:200]
                        ]
                    )
                    links.append(link)
                    
            except Exception as e:
                continue
        
        # Deduplicate by paper pair
        seen = set()
        unique_links = []
        for link in links:
            key = (tuple(link.source_papers), tuple(link.target_papers))
            if key not in seen:
                seen.add(key)
                unique_links.append(link)
        
        return unique_links[:top_k]
    
    def find_ecological_cascades(
        self,
        species_name: str,
        max_depth: int = 3
    ) -> list[CrossDomainLink]:
        """
        Find potential ecological cascade effects across domains.
        
        Traces ecological relationships through the graph to find
        how changes in one domain might affect another.
        """
        if not self.graph:
            return []
        
        links = []
        
        # Query: Find chains of ecological relationships
        query = """
        MATCH path = (s:Species {scientific_name: $name})-[:RELATES_TO*1..3]-(connected:Species)
        WITH path, [n IN nodes(path) | n.scientific_name] as species_chain
        MATCH (p:Paper)-[:MENTIONS]->(end_species:Species)
        WHERE end_species.scientific_name = last(species_chain)
          AND p.domain IS NOT NULL
        RETURN species_chain, 
               collect(DISTINCT p.domain) as domains,
               collect(DISTINCT p.doc_id) as papers
        LIMIT 20
        """
        
        try:
            with self.graph._driver.session(database=self.graph.database) as session:
                result = session.run(query, {"name": species_name})
                
                import hashlib
                for record in result:
                    chain = record["species_chain"]
                    domains = record["domains"]
                    papers = record["papers"]
                    
                    if len(domains) >= 2:
                        link_id = hashlib.md5(
                            ":".join(chain).encode()
                        ).hexdigest()[:12]
                        
                        # Get first and last domain
                        try:
                            source_d = DomainType(domains[0])
                            target_d = DomainType(domains[-1])
                        except ValueError:
                            continue
                        
                        link = CrossDomainLink(
                            link_id=link_id,
                            link_type=LinkType.ECOLOGICAL_CASCADE,
                            source_domain=source_d,
                            source_entity=chain[0],
                            source_papers=papers[:5],
                            target_domain=target_d,
                            target_entity=chain[-1],
                            target_papers=papers[-5:],
                            description=f"Ecological chain: {' → '.join(chain)}",
                            confidence=0.5,
                            shared_attributes={"chain": chain}
                        )
                        links.append(link)
                        
        except Exception as e:
            logger.error(f"Cascade query failed: {e}")
        
        return links
    
    def discover_all_links(
        self,
        limit_per_type: int = 10
    ) -> dict[LinkType, list[CrossDomainLink]]:
        """
        Run all link discovery methods and return organized results.
        """
        all_links = {lt: [] for lt in LinkType}
        
        # Get sample species from graph
        if self.graph:
            try:
                query = """
                MATCH (s:Species)<-[:MENTIONS]-(p:Paper)
                WHERE p.domain IS NOT NULL
                WITH s.scientific_name as species, count(DISTINCT p.domain) as domain_count
                WHERE domain_count >= 2
                RETURN species
                ORDER BY domain_count DESC
                LIMIT 20
                """
                with self.graph._driver.session(database=self.graph.database) as session:
                    result = session.run(query)
                    species_list = [r["species"] for r in result]
                    
                for species in species_list[:5]:
                    links = self.find_shared_species_links(species)
                    all_links[LinkType.SHARED_SPECIES].extend(links)
                    
                    cascade_links = self.find_ecological_cascades(species)
                    all_links[LinkType.ECOLOGICAL_CASCADE].extend(cascade_links)
                    
            except Exception as e:
                logger.error(f"Discovery failed: {e}")
        
        # Limit results
        for link_type in all_links:
            all_links[link_type] = all_links[link_type][:limit_per_type]
        
        return all_links
