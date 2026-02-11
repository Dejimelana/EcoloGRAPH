"""
NetworkX analytics for EcoloGRAPH.

Builds in-memory graphs from PaperIndex data and computes:
- Community detection (Girvan-Newman / greedy modularity)
- Centrality metrics (degree, betweenness)
- Contextual proximity (keyword co-occurrence across papers)

Adapted from rahulnyk/knowledge_graph's contextual_proximity() and
community detection approach.
"""
import logging
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)


# ── Graph Builders ──────────────────────────────────────────────

def build_paper_graph(papers, min_similarity: float = 0.55) -> nx.Graph:
    """
    Build a paper-to-paper graph based on domain similarity.

    Uses Jaccard similarity of domain sets (shared / union) so that papers
    with many domains don't automatically connect to everything.
    Threshold 0.55 means >55% domain overlap is needed.
    """
    G = nx.Graph()

    for p in papers:
        domain = (p.primary_domain or "unknown").replace("_", " ").title()
        short = (p.title[:40] + "…") if p.title and len(p.title) > 40 else (p.title or "?")
        G.add_node(
            p.doc_id,
            label=short,
            full_title=p.title or "?",
            year=p.year,
            domain=domain,
            primary_domain=p.primary_domain or "unknown",
            node_type="paper",
        )

    # Build edges from domain similarity (Jaccard)
    for i, p1 in enumerate(papers):
        dom1 = set(p1.domains.keys()) if p1.domains else set()
        kw1 = set(p1.keywords)

        for j in range(i + 1, len(papers)):
            p2 = papers[j]
            dom2 = set(p2.domains.keys()) if p2.domains else set()
            kw2 = set(p2.keywords)

            # Jaccard similarity for domains
            union = dom1 | dom2
            shared_dom = dom1 & dom2
            jaccard = len(shared_dom) / len(union) if union else 0

            # Bonus for shared keywords (if any exist)
            shared_kw = kw1 & kw2
            weight = jaccard + len(shared_kw) * 0.1

            if weight >= min_similarity:
                shared_labels = [d.replace('_', ' ') for d in sorted(shared_dom)[:4]]
                G.add_edge(
                    p1.doc_id, p2.doc_id,
                    weight=weight,
                    shared_domains=list(shared_dom),
                    title="Shared: " + ", ".join(shared_labels),
                )

    return G


def build_concept_graph(papers, min_cooccurrence: int = 2) -> nx.Graph:
    """
    Build a concept co-occurrence graph from domains and title words.

    Adapted from rahulnyk's contextual_proximity() algorithm:
    - For each paper, collect its concepts (domains + meaningful title words)
    - Self-join: all concept pairs within the same paper are connected
    - Group by pair, sum co-occurrences
    - Filter: keep pairs with co-occurrence >= min_cooccurrence

    Uses domains as primary concepts (since keywords are typically empty)
    and supplements with significant words from paper titles.
    """
    G = nx.Graph()
    STOP_WORDS = {
        'a', 'an', 'the', 'of', 'in', 'for', 'and', 'or', 'to', 'on',
        'with', 'by', 'from', 'as', 'at', 'its', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'this', 'that', 'it', 'not',
        'but', 'if', 'do', 'does', 'did', 'has', 'have', 'had',
        'using', 'based', 'new', 'novel', 'study', 'analysis',
        'approach', 'method', 'data', 'model', 'results', 'case',
        'between', 'through', 'during', 'after', 'before', 'under',
        'over', 'into', 'about', 'than', 'more', 'most', 'other',
        'how', 'what', 'which', 'each', 'can', 'may', 'could',
        'will', 'would', 'should', 'two', 'three', 'vs', 'via',
    }

    def _extract_concepts(paper):
        """Get concepts from a paper: domains + title words."""
        concepts = set()
        # Domains are our primary concepts
        if paper.domains:
            for domain in paper.domains.keys():
                concepts.add(domain.replace('_', ' '))
        # Keywords if available
        for kw in paper.keywords:
            kw_clean = kw.strip().lower()
            if len(kw_clean) > 2:
                concepts.add(kw_clean)
        # Title words (significant ones, lowered)
        if paper.title:
            words = paper.title.lower().split()
            for word in words:
                clean = word.strip('.,;:!?()[]"\'-–—')
                if len(clean) > 3 and clean not in STOP_WORDS:
                    concepts.add(clean)
        return concepts

    # Collect concept → paper_ids mapping
    concept_papers = defaultdict(set)
    for p in papers:
        for concept in _extract_concepts(p):
            concept_papers[concept].add(p.doc_id)

    # Self-join: for each paper, connect all concept pairs
    pair_count = defaultdict(int)
    pair_papers = defaultdict(set)

    for p in papers:
        concepts = sorted(_extract_concepts(p))
        for i, c1 in enumerate(concepts):
            for j in range(i + 1, len(concepts)):
                c2 = concepts[j]
                pair = (c1, c2)  # already sorted
                pair_count[pair] += 1
                pair_papers[pair].add(p.doc_id)

    # Build graph from pairs meeting threshold
    for (c1, c2), count in pair_count.items():
        if count >= min_cooccurrence:
            if c1 not in G:
                G.add_node(c1, label=c1.title(), node_type="concept",
                           paper_count=len(concept_papers[c1]))
            if c2 not in G:
                G.add_node(c2, label=c2.title(), node_type="concept",
                           paper_count=len(concept_papers[c2]))

            G.add_edge(c1, c2, weight=count,
                       title=f"Co-occur in {count} papers",
                       papers=list(pair_papers[(c1, c2)])[:5])

    return G


def build_domain_graph(papers) -> nx.Graph:
    """Build a domain-to-paper bipartite graph."""
    G = nx.Graph()

    domain_papers = defaultdict(list)
    for p in papers:
        if p.primary_domain:
            domain_papers[p.primary_domain].append(p)

    for domain, plist in domain_papers.items():
        domain_label = domain.replace("_", " ").title()
        G.add_node(
            f"d_{domain}", label=domain_label, node_type="domain",
            size=len(plist), domain=domain,
        )

        for p in plist:
            short = (p.title[:35] + "…") if p.title and len(p.title) > 35 else (p.title or "?")
            G.add_node(
                p.doc_id, label=short, full_title=p.title or "?",
                node_type="paper", domain=domain,
                year=p.year,
            )
            G.add_edge(f"d_{domain}", p.doc_id, weight=1.0)

    return G


# ── Analytics ───────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> dict[str, int]:
    """
    Detect communities using greedy modularity (fast, works on large graphs).
    Falls back to connected components if graph is too sparse.

    Returns: dict mapping node_id → community_index
    """
    if G.number_of_nodes() == 0:
        return {}

    try:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")
        mapping = {}
        for i, community in enumerate(communities):
            for node in community:
                mapping[node] = i
        return mapping
    except Exception:
        # Fallback: connected components
        mapping = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                mapping[node] = i
        return mapping


def compute_centrality(G: nx.Graph) -> dict[str, float]:
    """Compute degree centrality for node sizing."""
    if G.number_of_nodes() == 0:
        return {}
    return nx.degree_centrality(G)


# ── Color Palette ───────────────────────────────────────────────

COMMUNITY_COLORS = [
    "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
    "#14b8a6", "#a855f7", "#f43f5e", "#0ea5e9", "#22c55e",
    "#eab308", "#d946ef", "#64748b", "#fb923c", "#2dd4bf",
]

DOMAIN_COLORS = {
    "marine_ecology": "#06b6d4",
    "coral_reef_ecology": "#f97316",
    "freshwater_ecology": "#3b82f6",
    "conservation": "#10b981",
    "population_ecology": "#8b5cf6",
    "community_ecology": "#ec4899",
    "forest_ecology": "#84cc16",
    "climate_change_ecology": "#ef4444",
    "fisheries": "#06b6d4",
    "molecular_ecology": "#6366f1",
    "remote_sensing": "#0ea5e9",
    "deep_learning": "#a855f7",
    "species_distribution": "#14b8a6",
    "biodiversity": "#22c55e",
    "soil_ecology": "#eab308",
}


def community_color(community_id: int) -> str:
    """Get a color for a community index."""
    return COMMUNITY_COLORS[community_id % len(COMMUNITY_COLORS)]


def domain_color(domain: str) -> str:
    """Get a color for a domain name."""
    return DOMAIN_COLORS.get(domain, "#94a3b8")
