"""
Graph module - Neo4j knowledge graph + NetworkX analytics.

Provides:
- GraphBuilder: Build and manage the Neo4j knowledge graph
- Pre-built Cypher queries for analysis
- NetworkX graph analytics (community detection, centrality, co-occurrence)
- Graph builders for paper similarity, concept maps, and domain networks
"""
from .graph_builder import GraphBuilder, GraphStats
from .network_analysis import (
    build_paper_graph,
    build_concept_graph,
    build_domain_graph,
    detect_communities,
    compute_centrality,
    community_color,
)

__all__ = [
    "GraphBuilder",
    "GraphStats",
    "build_paper_graph",
    "build_concept_graph",
    "build_domain_graph",
    "detect_communities",
    "compute_centrality",
    "community_color",
]
