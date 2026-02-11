"""
Tool Registry for EcoloGRAPH Query Agent.

Defines tools that the LangGraph agent can invoke:
- search_papers: Hybrid BM25 + semantic search
- search_by_domain: Filter by scientific domain
- classify_text: Multi-domain classification
- get_species_info: External API lookups (FishBase, GBIF, IUCN)
- find_cross_domain_links: Cross-domain connection discovery
- generate_hypotheses: Scientific hypothesis generation
- query_graph: Cypher queries on Neo4j knowledge graph

Each tool is a LangChain-compatible tool using @tool decorator.
"""
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ============================================================
# Shared instances (lazy-loaded singletons)
# ============================================================

_paper_index = None
_ranked_search = None
_domain_classifier = None
_cross_domain_linker = None
_inference_proposer = None
_graph_builder = None


def _get_paper_index():
    """Lazy-load PaperIndex."""
    global _paper_index
    if _paper_index is None:
        from ..search.paper_index import PaperIndex
        _paper_index = PaperIndex()
    return _paper_index


def _get_ranked_search():
    """Lazy-load RankedSearch."""
    global _ranked_search
    if _ranked_search is None:
        from ..search.ranked_search import RankedSearch
        _ranked_search = RankedSearch(paper_index=_get_paper_index())
    return _ranked_search


def _get_classifier():
    """Lazy-load DomainClassifier."""
    global _domain_classifier
    if _domain_classifier is None:
        from ..extraction.domain_classifier import DomainClassifier
        _domain_classifier = DomainClassifier()
    return _domain_classifier


def _get_linker():
    """Lazy-load CrossDomainLinker."""
    global _cross_domain_linker
    if _cross_domain_linker is None:
        from ..inference.cross_domain_linker import CrossDomainLinker
        _cross_domain_linker = CrossDomainLinker()
    return _cross_domain_linker


def _get_proposer():
    """Lazy-load InferenceProposer."""
    global _inference_proposer
    if _inference_proposer is None:
        from ..inference.inference_proposer import InferenceProposer
        _inference_proposer = InferenceProposer()
    return _inference_proposer


def _get_graph_builder():
    """Lazy-load GraphBuilder (returns None if Neo4j unreachable)."""
    global _graph_builder
    if _graph_builder is None:
        try:
            from ..graph.graph_builder import GraphBuilder
            _graph_builder = GraphBuilder()
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            return None
    return _graph_builder


# ============================================================
# Tool Definitions
# ============================================================

@tool
def search_papers(query: str, limit: int = 10) -> str:
    """
    Search scientific papers by query. Uses hybrid BM25 keyword + semantic ranking.
    Use this for general questions about ecology, species, or any scientific topic.
    
    Args:
        query: Natural language search query (e.g., 'microplastics marine fish')
        limit: Maximum number of results to return
    """
    try:
        search = _get_ranked_search()
        results = search.search(query=query, limit=limit)
        
        if not results:
            return f"No papers found for query: '{query}'"
        
        output_lines = [f"Found {len(results)} papers for '{query}':\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(
                f"[{i}] {r.title} (score: {r.combined_score:.3f})"
            )
            if r.primary_domain:
                output_lines.append(f"    Domain: {r.primary_domain}")
            if r.year:
                output_lines.append(f"    Year: {r.year}")
            if r.snippet:
                output_lines.append(f"    Snippet: {r.snippet[:150]}...")
            output_lines.append("")
        
        return "\n".join(output_lines)
    except Exception as e:
        logger.error(f"search_papers failed: {e}")
        return f"Search error: {e}"


@tool
def search_by_domain(query: str, domain: str, limit: int = 10) -> str:
    """
    Search papers within a specific scientific domain.
    Use this when the user asks about a particular field of study.
    
    Args:
        query: Natural language search query
        domain: Scientific domain (e.g., 'marine_ecology', 'conservation', 'genetics',
                'machine_learning', 'soundscape_ecology', 'ethology', 'biogeography')
        limit: Maximum number of results
    """
    try:
        search = _get_ranked_search()
        results = search.search_by_domain(query=query, domain=domain, limit=limit)
        
        if not results:
            return f"No papers found for '{query}' in domain '{domain}'"
        
        output_lines = [f"Found {len(results)} papers for '{query}' in {domain}:\n"]
        for i, r in enumerate(results, 1):
            output_lines.append(f"[{i}] {r.title} (score: {r.combined_score:.3f})")
            if r.snippet:
                output_lines.append(f"    {r.snippet[:150]}...")
            output_lines.append("")
        
        return "\n".join(output_lines)
    except Exception as e:
        logger.error(f"search_by_domain failed: {e}")
        return f"Search error: {e}"


@tool
def classify_text(text: str) -> str:
    """
    Classify a text into scientific domains. Returns primary domain and secondary domains.
    Use this to understand what scientific areas a piece of text belongs to.
    
    Args:
        text: Text to classify (abstract, paragraph, or full document text)
    """
    try:
        classifier = _get_classifier()
        
        # Call classify_text directly (not classify_document which needs ParsedDocument)
        result = classifier.classify_text(text, use_llm=False)
        
        lines = [
            f"Primary domain: {result.primary_domain.value} ({result.confidence:.1%})",
            f"Study type: {result.study_type.value}",
            f"Method: {result.method}",
            "\nDomain scores:"
        ]
        
        # get_top_domains is on DomainClassifier, not ClassificationResult
        top_domains = classifier.get_top_domains(result, threshold=0.05)
        for dt, score in top_domains[:5]:
            bar = "█" * int(score * 20)
            lines.append(f"  {dt.value:25} {score:6.1%} {bar}")
        
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"classify_text failed: {e}")
        return f"Classification error: {e}"


@tool
def get_species_info(species_name: str) -> str:
    """
    Get ecological information about a species from external databases.
    Queries FishBase (fish), GBIF (distribution), and IUCN (conservation status).
    
    Args:
        species_name: Scientific name of the species (e.g., 'Gadus morhua')
    """
    results = []
    
    # Try FishBase
    try:
        from ..scrapers.fishbase_client import FishBaseClient
        with FishBaseClient() as fb:
            info = fb.get_species(species_name)
            if info:
                results.append("=== FishBase ===")
                if info.fb_name:
                    results.append(f"Common name: {info.fb_name}")
                elif info.common_names:
                    results.append(f"Common name: {info.common_names[0].get('name', 'N/A')}")
                if info.genus and info.species:
                    results.append(f"Taxonomy: {info.genus} {info.species}")
                if info.habitat:
                    results.append(f"Habitat: {info.habitat}")
                if info.climate:
                    results.append(f"Climate: {info.climate}")
                if info.max_length_cm:
                    results.append(f"Max length: {info.max_length_cm} cm")
                if info.max_weight_g:
                    results.append(f"Max weight: {info.max_weight_g} g")
                if info.trophic_level:
                    results.append(f"Trophic level: {info.trophic_level:.1f}")
                if info.feeding_type:
                    results.append(f"Feeding: {info.feeding_type}")
                if info.depth_range_shallow is not None and info.depth_range_deep is not None:
                    results.append(f"Depth range: {info.depth_range_shallow}-{info.depth_range_deep} m")
                if info.iucn_status:
                    results.append(f"IUCN (via FishBase): {info.iucn_status}")
    except Exception as e:
        logger.debug(f"FishBase lookup failed: {e}")
    
    # Try GBIF
    try:
        from ..scrapers.gbif_occurrence_client import GBIFOccurrenceClient
        with GBIFOccurrenceClient() as gbif:
            dist = gbif.get_distribution(species_name, sample_size=5)
            if dist and dist.total_occurrences > 0:
                results.append(f"\n=== GBIF ({dist.total_occurrences:,} occurrences) ===")
                if dist.countries:
                    results.append(f"Countries: {', '.join(dist.countries[:10])}")
                if dist.min_year and dist.max_year:
                    results.append(f"Records span: {dist.min_year}-{dist.max_year}")
                if dist.occurrences_with_coords:
                    results.append(f"Georeferenced: {dist.occurrences_with_coords:,}")
    except Exception as e:
        logger.debug(f"GBIF lookup failed: {e}")
    
    # Try IUCN
    try:
        from ..scrapers.iucn_client import IUCNClient
        with IUCNClient() as iucn:
            status = iucn.get_species(species_name)
            if status:
                results.append(f"\n=== IUCN Red List ===")
                if status.category and status.category_name:
                    results.append(f"Status: {status.category} ({status.category_name})")
                elif status.category:
                    results.append(f"Status: {status.category}")
                if status.population_trend:
                    results.append(f"Population trend: {status.population_trend}")
                if status.threats:
                    results.append(f"Threats: {', '.join(status.threats[:5])}")
                if status.habitats:
                    results.append(f"Habitats: {', '.join(status.habitats[:5])}")
    except Exception as e:
        logger.debug(f"IUCN lookup failed: {e}")
    
    if results:
        return f"Species info for '{species_name}':\n" + "\n".join(results)
    return f"No information found for species '{species_name}'"


@tool
def find_cross_domain_links(domain1: str, domain2: str) -> str:
    """
    Find connections between two scientific domains.
    Discovers shared species, methods, or concepts across domains.
    
    Args:
        domain1: First domain (e.g., 'marine_ecology')
        domain2: Second domain (e.g., 'toxicology')
    """
    try:
        from ..core.domain_registry import DomainType
        
        dt1 = DomainType(domain1)
        dt2 = DomainType(domain2)
        
        linker = _get_linker()
        affinity = linker.get_domain_affinity(dt1, dt2)
        
        lines = [
            f"Cross-domain analysis: {domain1} ↔ {domain2}",
            f"Affinity score: {affinity:.2f}",
        ]
        
        if affinity >= 0.8:
            lines.append("Relationship: STRONG - These domains frequently co-occur")
        elif affinity >= 0.5:
            lines.append("Relationship: MODERATE - Some connections expected")
        else:
            lines.append("Relationship: WEAK - Limited direct connections")
        
        return "\n".join(lines)
    except ValueError as e:
        return f"Invalid domain name: {e}"
    except Exception as e:
        logger.error(f"find_cross_domain_links failed: {e}")
        return f"Error: {e}"


@tool
def generate_hypotheses(topic: str, domains: str = "") -> str:
    """
    Generate scientific research hypotheses about a topic.
    Can focus on specific domain intersections.
    
    Args:
        topic: Research topic (e.g., 'effect of ocean acidification on coral reefs')
        domains: Comma-separated domains to focus on (e.g., 'marine_ecology,conservation')
    """
    try:
        from ..core.domain_registry import DomainType
        from ..inference.cross_domain_linker import CrossDomainLink, LinkType
        import hashlib
        
        domain_list = [d.strip() for d in domains.split(",") if d.strip()] if domains else []
        
        # Build CrossDomainLink objects from domains (the real API needs these)
        links = []
        if len(domain_list) >= 2:
            # Create links between each pair of domains
            for i in range(len(domain_list)):
                for j in range(i + 1, len(domain_list)):
                    try:
                        dt1 = DomainType(domain_list[i])
                        dt2 = DomainType(domain_list[j])
                        linker = _get_linker()
                        affinity = linker.get_domain_affinity(dt1, dt2)
                        
                        link_id = hashlib.md5(
                            f"{domain_list[i]}:{domain_list[j]}:{topic}".encode()
                        ).hexdigest()[:12]
                        
                        link = CrossDomainLink(
                            link_id=link_id,
                            link_type=LinkType.COMPLEMENTARY_DATA,
                            source_domain=dt1,
                            source_entity=topic,
                            target_domain=dt2,
                            target_entity=topic,
                            confidence=affinity,
                            description=f"Cross-domain link for: {topic}",
                        )
                        links.append(link)
                    except ValueError:
                        continue
        else:
            # Single domain or no domain: create an exploratory link
            dt1 = DomainType(domain_list[0]) if domain_list else DomainType.GENERAL_ECOLOGY
            dt2 = DomainType.CONSERVATION if dt1 != DomainType.CONSERVATION else DomainType.MARINE_ECOLOGY
            
            link_id = hashlib.md5(f"explore:{topic}".encode()).hexdigest()[:12]
            link = CrossDomainLink(
                link_id=link_id,
                link_type=LinkType.COMPLEMENTARY_DATA,
                source_domain=dt1,
                source_entity=topic,
                target_domain=dt2,
                target_entity=topic,
                confidence=0.5,
                description=f"Exploratory hypothesis for: {topic}",
            )
            links.append(link)
        
        # Generate hypotheses using real API
        proposer = _get_proposer()
        hypotheses = []
        for link in links:
            try:
                h = proposer._generate_rule_based_hypothesis(link)
                hypotheses.append(h)
            except Exception as e:
                logger.debug(f"Hypothesis generation failed for link {link.link_id}: {e}")
        
        if not hypotheses:
            return f"Could not generate hypotheses for topic: '{topic}'"
        
        lines = [f"Hypotheses for '{topic}':\n"]
        for i, h in enumerate(hypotheses, 1):
            lines.append(f"[H{i}] {h.statement}")
            lines.append(f"     Type: {h.hypothesis_type.value}")
            lines.append(f"     Confidence: {h.confidence.value} ({h.confidence_score:.1%})")
            if h.rationale:
                lines.append(f"     Rationale: {h.rationale}")
            if h.suggested_experiments:
                lines.append(f"     Experiments: {'; '.join(h.suggested_experiments[:2])}")
            lines.append("")
        
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"generate_hypotheses failed: {e}")
        return f"Error generating hypotheses: {e}"


@tool
def query_graph(species_name: str, query_type: str = "profile") -> str:
    """
    Query the Neo4j knowledge graph for ecological data.
    Use this when asked about relationships, networks, or data synthesis across papers.
    
    Args:
        species_name: Scientific name to query (e.g., 'Gadus morhua')
        query_type: Type of query - one of: 'profile' (full species info),
                    'network' (ecological relationships), 'measurements' (quantitative data),
                    'papers' (publications mentioning species), 'co_occurrence' (co-occurring species)
    """
    try:
        gb = _get_graph_builder()
        if gb is None:
            return "Graph database (Neo4j) is not available. Start Neo4j to use graph queries."
        
        if query_type == "profile":
            from ..graph import queries
            result = gb._driver.session().run(
                queries.SPECIES_PROFILE, name=species_name
            )
            records = [dict(r) for r in result]
            if not records:
                return f"No profile found for '{species_name}' in the knowledge graph."
            profile = records[0].get("profile", {})
            lines = [f"Species Profile: {species_name}"]
            if profile.get("family"):
                lines.append(f"  Family: {profile['family']}")
            papers = profile.get("papers", [])
            if papers:
                lines.append(f"  Papers: {len(papers)} publications")
                for p in papers[:5]:
                    lines.append(f"    - {p.get('title', 'N/A')} ({p.get('year', '?')})")
            measurements = profile.get("measurements", [])
            if measurements:
                lines.append(f"  Measurements: {len(measurements)}")
                for m in measurements[:5]:
                    lines.append(f"    - {m.get('parameter', '?')}: {m.get('value', '?')} {m.get('unit', '')}")
            return "\n".join(lines)
        
        elif query_type == "network":
            results = gb.get_ecological_network(species_name, depth=2)
            if not results:
                return f"No ecological network found for '{species_name}'."
            lines = [f"Ecological Network for {species_name}:"]
            for r in results[:15]:
                lines.append(f"  {r.get('from_species', '?')} --[{r.get('relation_type', '?')}]--> {r.get('to_species', '?')}")
            return "\n".join(lines)
        
        elif query_type == "measurements":
            results = gb.get_species_measurements(species_name)
            if not results:
                return f"No measurements found for '{species_name}'."
            lines = [f"Measurements for {species_name}:"]
            for r in results[:20]:
                lines.append(f"  {r.get('parameter', '?')}: {r.get('value', '?')} {r.get('unit', '')}")
            return "\n".join(lines)
        
        elif query_type == "papers":
            results = gb.get_species_papers(species_name)
            if not results:
                return f"No papers found mentioning '{species_name}'."
            lines = [f"Papers mentioning {species_name}: {len(results)}"]
            for r in results[:10]:
                lines.append(f"  - {r.get('title', 'N/A')} ({r.get('year', '?')})")
            return "\n".join(lines)
        
        elif query_type == "co_occurrence":
            from ..graph import queries
            result = gb._driver.session().run(
                queries.SPECIES_CO_OCCURRENCE, name=species_name
            )
            records = [dict(r) for r in result]
            if not records:
                return f"No co-occurring species found for '{species_name}'."
            lines = [f"Species co-occurring with {species_name}:"]
            for r in records[:15]:
                lines.append(f"  {r.get('co_occurring_species', '?')} ({r.get('family', '?')}) - {r.get('shared_papers', 0)} shared papers")
            return "\n".join(lines)
        
        else:
            return f"Unknown query_type '{query_type}'. Use: profile, network, measurements, papers, co_occurrence"
    
    except Exception as e:
        logger.error(f"query_graph failed: {e}")
        return f"Graph query error: {e}"


@tool
def search_related_papers(doc_id: str, limit: int = 5) -> str:
    """
    Find papers related to a given paper using the knowledge graph.
    Uses domain similarity and co-occurrence to find connected papers.
    Use this when asked about papers 'related to', 'similar to', or 'connected to' another paper.
    
    Args:
        doc_id: Document ID of the reference paper
        limit: Maximum number of related papers to return
    """
    try:
        idx = _get_paper_index()
        papers = idx.get_all_papers(limit=200)
        
        from src.graph.network_analysis import build_paper_graph
        G = build_paper_graph(papers, min_similarity=0.4)
        
        if doc_id not in G:
            return f"Paper '{doc_id}' not found in graph."
        
        # Get neighbors sorted by edge weight
        neighbors = []
        for neighbor in G.neighbors(doc_id):
            edge_data = G[doc_id][neighbor]
            node_data = G.nodes[neighbor]
            neighbors.append({
                "doc_id": neighbor,
                "title": node_data.get("full_title", "?"),
                "domain": node_data.get("domain", "?"),
                "similarity": round(edge_data.get("weight", 0), 3),
                "shared": ", ".join(
                    d.replace("_", " ") for d in edge_data.get("shared_domains", [])[:4]
                ),
            })
        
        neighbors.sort(key=lambda x: x["similarity"], reverse=True)
        neighbors = neighbors[:limit]
        
        if not neighbors:
            return f"No related papers found for '{doc_id}'."
        
        # Format
        ref_title = G.nodes[doc_id].get("full_title", doc_id)
        lines = [f"Papers related to '{ref_title}':\n"]
        for i, n in enumerate(neighbors, 1):
            lines.append(
                f"{i}. {n['title']}\n"
                f"   Domain: {n['domain']} | Similarity: {n['similarity']}\n"
                f"   Shared: {n['shared']}"
            )
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"search_related_papers failed: {e}")
        return f"Error: {e}"


# ============================================================
# All tools list (for agent registration)
# ============================================================

ALL_TOOLS = [
    search_papers,
    search_by_domain,
    classify_text,
    get_species_info,
    find_cross_domain_links,
    generate_hypotheses,
    query_graph,
    search_related_papers,
]


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools."""
    lines = ["Available tools:"]
    for t in ALL_TOOLS:
        lines.append(f"  • {t.name}: {t.description.split(chr(10))[0].strip()}")
    return "\n".join(lines)
