"""
Neo4j Graph Data Science (GDS) analytics wrapper.

Provides scalable graph analytics for EcoloGRAPH using Neo4j GDS library
instead of NetworkX when dealing with >1k papers.

Requires:
    - Neo4j 5.x with GDS plugin installed
    - graphdatascience Python library

Usage:
    analytics = Neo4jAnalytics(uri, user, password)
    analytics.create_paper_projection(min_similarity=0.55)
    communities = analytics.detect_communities()
    centrality = analytics.compute_pagerank()
"""
import logging
from typing import Any

try:
    from graphdatascience import GraphDataScience
except ImportError:
    GraphDataScience = None

logger = logging.getLogger(__name__)


class Neo4jAnalytics:
    """
    Wrapper for Neo4j Graph Data Science operations.
    
    Enables scalable community detection, centrality computation,
    and similarity analysis on large paper graphs (>1k papers).
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize GDS client.
        
        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            
        Raises:
            ImportError: If graphdatascience library not installed
            ConnectionError: If cannot connect to Neo4j
        """
        if GraphDataScience is None:
            raise ImportError(
                "graphdatascience library not installed. "
                "Install with: pip install graphdatascience"
            )
        
        try:
            self.gds = GraphDataScience(uri, auth=(user, password))
            logger.info(f"Connected to Neo4j GDS at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j GDS: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j at {uri}: {e}")
    
    def create_paper_projection(
        self, 
        min_similarity: float = 0.55,
        projection_name: str = "paper_graph"
    ) -> dict[str, Any]:
        """
        Create in-memory graph projection from Neo4j papers.
        
        Papers are connected if they share domains (Jaccard similarity).
        
        Args:
            min_similarity: Minimum Jaccard similarity (0-1) to create edge
            projection_name: Name for the projection
            
        Returns:
            Projection metadata (node count, relationship count, etc.)
        """
        # Drop existing projection if exists
        try:
            self.gds.graph.drop(projection_name)
            logger.info(f"Dropped existing projection '{projection_name}'")
        except Exception:
            pass  # Projection didn't exist
        
        # Create new projection using Cypher
        result, meta = self.gds.graph.project.cypher(
            projection_name,
            # Node query - all papers with metadata
            """
            MATCH (p:Paper)
            RETURN 
                id(p) AS id, 
                p.doc_id AS doc_id, 
                p.title AS title,
                p.primary_domain AS domain
            """,
            # Relationship query - papers sharing domains
            """
            MATCH (p1:Paper)-[:CLASSIFIED_AS]->(d:Domain)<-[:CLASSIFIED_AS]-(p2:Paper)
            WHERE id(p1) < id(p2)
            WITH p1, p2, 
                 COUNT(d) AS shared,
                 SIZE([(p1)-[:CLASSIFIED_AS]->() | 1]) AS p1_domains,
                 SIZE([(p2)-[:CLASSIFIED_AS]->() | 1]) AS p2_domains
            WITH p1, p2, 
                 toFloat(shared) / (p1_domains + p2_domains - shared) AS jaccard
            WHERE jaccard >= $min_similarity
            RETURN 
                id(p1) AS source, 
                id(p2) AS target, 
                jaccard AS weight
            """,
            parameters={"min_similarity": min_similarity}
        )
        
        logger.info(
            f"Created projection '{projection_name}': "
            f"{result['nodeCount']} nodes, {result['relationshipCount']} relationships"
        )
        
        return result
    
    def detect_communities(
        self, 
        projection_name: str = "paper_graph",
        algorithm: str = "louvain"
    ) -> dict[int, int]:
        """
        Detect research communities using Louvain algorithm.
        
        Args:
            projection_name: Name of graph projection
            algorithm: Algorithm to use ('louvain' or 'label_propagation')
            
        Returns:
            Dictionary mapping Neo4j node ID to community ID
        """
        if algorithm == "louvain":
            result = self.gds.louvain.stream(
                projection_name,
                relationshipWeightProperty="weight",
                includeIntermediateCommunities=False
            )
        elif algorithm == "label_propagation":
            result = self.gds.labelPropagation.stream(
                projection_name,
                relationshipWeightProperty="weight"
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        communities = {}
        for row in result:
            communities[row["nodeId"]] = row["communityId"]
        
        logger.info(f"Detected {len(set(communities.values()))} communities")
        
        return communities
    
    def compute_pagerank(
        self, 
        projection_name: str = "paper_graph",
        damping_factor: float = 0.85
    ) -> dict[int, float]:
        """
        Compute PageRank centrality for papers.
        
        Identifies influential papers based on citation-like structure.
        
        Args:
            projection_name: Name of graph projection
            damping_factor: PageRank damping factor (0-1)
            
        Returns:
            Dictionary mapping Neo4j node ID to PageRank score
        """
        result = self.gds.pageRank.stream(
            projection_name,
            relationshipWeightProperty="weight",
            dampingFactor=damping_factor
        )
        
        scores = {}
        for row in result:
            scores[row["nodeId"]] = row["score"]
        
        logger.info(f"Computed PageRank for {len(scores)} papers")
        
        return scores
    
    def compute_betweenness(
        self, 
        projection_name: str = "paper_graph"
    ) -> dict[int, float]:
        """
        Compute betweenness centrality.
        
        Identifies papers that bridge different research areas.
        
        Args:
            projection_name: Name of graph projection
            
        Returns:
            Dictionary mapping Neo4j node ID to betweenness score
        """
        result = self.gds.betweenness.stream(projection_name)
        
        scores = {}
        for row in result:
            scores[row["nodeId"]] = row["score"]
        
        logger.info(f"Computed betweenness for {len(scores)} papers")
        
        return scores
    
    def find_similar_papers(
        self,
        node_id: int,
        projection_name: str = "paper_graph",
        limit: int = 10
    ) -> list[tuple[int, float]]:
        """
        Find papers most similar to a given paper.
        
        Args:
            node_id: Neo4j node ID of reference paper
            projection_name: Name of graph projection
            limit: Maximum number of similar papers to return
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        # Use node similarity algorithm
        result = self.gds.nodeSimilarity.stream(
            projection_name,
            topK=limit,
            similarityCutoff=0.0
        )
        
        similar = []
        for row in result:
            if row["node1"] == node_id:
                similar.append((row["node2"], row["similarity"]))
            elif row["node2"] == node_id:
                similar.append((row["node1"], row["similarity"]))
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar[:limit]
    
    def drop_projection(self, projection_name: str = "paper_graph"):
        """
        Drop in-memory graph projection.
        
        Args:
            projection_name: Name of projection to drop
        """
        try:
            self.gds.graph.drop(projection_name)
            logger.info(f"Dropped projection '{projection_name}'")
        except Exception as e:
            logger.warning(f"Could not drop projection '{projection_name}': {e}")
    
    def close(self):
        """Close GDS connection."""
        if hasattr(self, 'gds'):
            self.gds.close()
            logger.info("Closed Neo4j GDS connection")
