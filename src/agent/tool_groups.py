"""
Tool grouping strategy for hierarchical routing.

Groups tools by research strategy:
- SEARCH_TOOLS: Local paper search and classification
- GRAPH_TOOLS: Neo4j knowledge graph queries
- EXTERNAL_TOOLS: External API lookups (GBIF, FishBase, etc.)
- INFERENCE_TOOLS: Cross-domain linkage and hypothesis generation

Tier 2 router classifies research intent and limits tool exposure.
"""

# Tool groups by strategy
from .tool_registry import (
    search_papers,
    search_by_domain,
    search_related_papers,
    classify_text,
    query_graph,
    get_species_info,
    find_cross_domain_links,
    generate_hypotheses,
)

# Local search & classification
SEARCH_TOOLS = [
    search_papers,
    search_by_domain,
    search_related_papers,
    classify_text,
]

# Knowledge graph queries
GRAPH_TOOLS = [
    query_graph,
    search_related_papers,  # Uses graph relationships
]

# External APIs
EXTERNAL_TOOLS = [
    get_species_info,
]

# Inference & cross-domain
INFERENCE_TOOLS = [
    find_cross_domain_links,
    generate_hypotheses,
]

# Strategy combinations
SEARCH_FOCUSED = SEARCH_TOOLS
GRAPH_FOCUSED = GRAPH_TOOLS + SEARCH_TOOLS[:2]  # graph + basic search
EXTERNAL_FOCUSED = EXTERNAL_TOOLS + SEARCH_TOOLS[:2]  # external + basic search
INFERENCE_FOCUSED = INFERENCE_TOOLS + SEARCH_TOOLS + GRAPH_TOOLS

# Mapping for router
STRATEGY_TO_TOOLS = {
    "search": SEARCH_FOCUSED,
    "graph": GRAPH_FOCUSED,
    "external": EXTERNAL_FOCUSED,
    "inference": INFERENCE_FOCUSED,
    "full": SEARCH_TOOLS + GRAPH_TOOLS + EXTERNAL_TOOLS + INFERENCE_TOOLS,
}


def get_tools_for_strategy(strategy: str) -> list:
    """Get tool subset for a given research strategy."""
    return STRATEGY_TO_TOOLS.get(strategy, STRATEGY_TO_TOOLS["full"])
