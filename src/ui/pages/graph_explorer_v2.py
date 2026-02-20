"""
EcoloGRAPH — Interactive Graph Explorer V2.

Connected Papers-style interface with:
- Click-to-explore nodes
- Sidebar with metadata
- Chunk viewer
- Density controls
- Multiple layout options
"""
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from src.ui.theme import inject_css
from src.graph.graph_builder import GraphBuilder
from src.search.paper_index import PaperIndex
import logging

logger = logging.getLogger(__name__)


def render():
    """Main render function for Graph Explorer V2."""
    inject_css()
    
    st.markdown(
        '<div class="hero-title">🕸️ Interactive Graph Explorer</div>'
        '<div class="hero-subtitle">Click nodes to explore • Scroll down for details & source chunks</div>',
        unsafe_allow_html=True,
    )
    
    # Initialize session state
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "selected_node_type" not in st.session_state:
        st.session_state.selected_node_type = None
    
    # Check data availability
    papers, graph_available = _check_data_sources()
    
    if not papers:
        st.info(
            "No papers indexed yet. Run the ingestion pipeline first:\n\n"
            "```\npython scripts/ingest.py data/papers/\n```"
        )
        return
    
    # --- View Mode Tabs ---
    tabs = st.tabs([
        "🎯 Explorer", "🔬 Species", "🏷️ Domains", "📄 Papers",
        "🔧 Methodology", "👤 Authors", "📍 Locations"
    ])
    
    with tabs[0]:  # Explorer (existing)
        if not graph_available:
            st.warning("⚠️ Neo4j not available. Some features will be limited.")
        else:
            _render_graph_panel(graph_available)
            if st.session_state.selected_node:
                st.markdown("---")
                _render_node_details_panel()
    
    with tabs[1]:  # Species
        _render_tab_species(graph_available)
    
    with tabs[2]:  # Domains
        _render_tab_domains(papers)
    
    with tabs[3]:  # Papers
        _render_tab_papers(papers, graph_available)
    
    with tabs[4]:  # Methodology 
        _render_tab_methodology(papers)
    
    with tabs[5]:  # Authors
        _render_tab_authors(papers)
    
    with tabs[6]:  # Locations
        _render_tab_locations(graph_available)


def _check_data_sources():
    """Check if papers and Neo4j are available."""
    papers = []
    graph_available = False
    
    try:
        idx = PaperIndex()
        papers = idx.get_all_papers(limit=200)
        logger.info(f"Loaded {len(papers)} papers from SQLite")
    except Exception as e:
        logger.error(f"Failed to load papers: {e}")
    
    try:
        gb = GraphBuilder()
        stats = gb.get_stats()
        graph_available = stats.paper_count > 0
        gb.close()
        logger.info(f"Neo4j available: {graph_available}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
    
    return papers, graph_available


def _render_graph_panel(graph_available):
    """Render the main graph visualization panel."""
    st.markdown("### 🎛️ Graph Controls")
    
    # Controls row 1: Layout and filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        layout_mode = st.selectbox(
            "Layout",
            [
                "Force-Directed", "Hierarchical (Top-Down)",
                "Hierarchical (Left-Right)", "Repulsion",
                "ForceAtlas2"
            ],
            index=0,
            help=(
                "Force-Directed: barnesHut physics • "
                "Hierarchical: tree layouts (top-down or left-right) • "
                "Repulsion: strong node separation • "
                "ForceAtlas2: community clustering"
            )
        )
    
    with col2:
        min_connections = st.slider(
            "Min Connections",
            min_value=0,
            max_value=10,
            value=0,
            help="Hide nodes with fewer connections"
        )
    
    with col3:
        max_nodes = st.slider(
            "Max Nodes",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Limit graph size for performance"
        )
    
    with col4:
        domain_filter = st.multiselect(
            "Domains",
            ["All", "ecology", "marine_biology", "climate_science", "remote_sensing"],
            default=["All"]
        )
    
    # Controls row 2: Visual options
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        show_labels = st.checkbox("Show Labels", value=True)
    
    with col_v2:
        search_node = st.text_input(
            "🔍 Find Node",
            placeholder="Search by title or species...",
            help="Highlight nodes matching your search"
        )
    
    # Physics settings expander (not for hierarchical)
    is_hierarchical = layout_mode.startswith("Hierarchical")
    
    if not is_hierarchical:
        with st.expander("⚙️ Physics Settings", expanded=False):
            pcol1, pcol2, pcol3 = st.columns(3)
            
            with pcol1:
                gravity = st.slider(
                    "Gravity", -80000, -1000,
                    -50000 if layout_mode == "Repulsion" else -30000,
                    1000
                )
                spring_length = st.slider("Spring Length", 50, 500, 200, 10)
            
            with pcol2:
                spring_constant = st.slider("Spring Constant", 0.01, 0.20, 0.04, 0.01)
                damping = st.slider("Damping", 0.01, 0.50, 0.09, 0.01)
            
            with pcol3:
                node_distance = st.slider("Node Distance", 50, 300, 120, 10)
                central_gravity = st.slider("Central Gravity", 0.0, 1.0, 0.3, 0.05)
    
    st.markdown("---")
    
    # Build and render graph
    try:
        nodes, edges = _build_graph_data(
            min_connections=min_connections,
            domain_filter=domain_filter if "All" not in domain_filter else None,
            max_nodes=max_nodes
        )
        
        if not nodes:
            st.warning("No nodes to display with current filters.")
            return
        
        # Apply search highlight
        if search_node:
            search_lower = search_node.lower()
            for node in nodes:
                if search_lower in (node.label or "").lower():
                    node.color = "#FF6B6B"
                    node.size = 30
        
        # Build vis.js-compatible physics configuration
        if is_hierarchical:
            config = Config(
                width="100%",
                height=600,
                directed=False,
                physics=False,
                hierarchical=True,
                nodeHighlightBehavior=True,
                highlightColor="#FF6B6B",
                collapsible=True,
                node={'labelProperty': 'label' if show_labels else None},
                link={'labelProperty': 'label', 'renderLabel': False}
            )
        elif layout_mode == "Repulsion":
            config = Config(
                width="100%",
                height=600,
                directed=False,
                physics={
                    "enabled": True,
                    "solver": "repulsion",
                    "repulsion": {
                        "centralGravity": central_gravity,
                        "springLength": spring_length,
                        "springConstant": spring_constant,
                        "nodeDistance": node_distance,
                        "damping": damping
                    },
                    "stabilization": {"iterations": 150}
                },
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#FF6B6B",
                collapsible=True,
                node={'labelProperty': 'label' if show_labels else None},
                link={'labelProperty': 'label', 'renderLabel': False}
            )
        elif layout_mode == "ForceAtlas2":
            config = Config(
                width="100%",
                height=600,
                directed=False,
                physics={
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": gravity,
                        "centralGravity": central_gravity,
                        "springLength": spring_length,
                        "springConstant": spring_constant,
                        "damping": damping,
                        "avoidOverlap": 0.5
                    },
                    "stabilization": {"iterations": 150}
                },
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#FF6B6B",
                collapsible=True,
                node={'labelProperty': 'label' if show_labels else None},
                link={'labelProperty': 'label', 'renderLabel': False}
            )
        else:  # Force-Directed (barnesHut default)
            config = Config(
                width="100%",
                height=600,
                directed=False,
                physics={
                    "enabled": True,
                    "solver": "barnesHut",
                    "barnesHut": {
                        "gravitationalConstant": gravity,
                        "centralGravity": central_gravity,
                        "springLength": spring_length,
                        "springConstant": spring_constant,
                        "damping": damping,
                        "avoidOverlap": 0
                    },
                    "stabilization": {"iterations": 150}
                },
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#FF6B6B",
                collapsible=True,
                node={'labelProperty': 'label' if show_labels else None},
                link={'labelProperty': 'label', 'renderLabel': False}
            )
        
        # Render graph with click callback
        selected = agraph(nodes=nodes, edges=edges, config=config)
        
        # Handle node selection
        if selected:
            st.session_state.selected_node = selected
            _handle_node_click(selected)
            
    except Exception as e:
        st.error(f"Failed to render graph: {e}")
        logger.exception("Graph rendering error")


def _build_graph_data(min_connections=0, domain_filter=None, max_nodes=200):
    """
    Build nodes and edges for agraph.
    
    Strategy:
    1. Try Neo4j first (if entities extracted)
    2. Fallback to PaperIndex (domain-based graph if no entities)
    
    Args:
        min_connections: Minimum connection count filter
        domain_filter: Optional domain filter
        max_nodes: Maximum number of paper nodes to load
    
    Returns:
        Tuple of (nodes, edges) lists
    """
    nodes = []
    edges = []
    
    # Try Neo4j first (requires entity extraction)
    try:
        gb = GraphBuilder()
        
        query = """
        MATCH (p:Paper)-[r:MENTIONS]->(s:Species)
        WITH p, s, count(r) as mention_count
        WHERE mention_count >= $min_connections
        RETURN 
            p.doc_id as paper_id,
            p.title as paper_title,
            p.year as paper_year,
            collect(DISTINCT s.scientific_name) as species
        ORDER BY mention_count DESC
        LIMIT $max_nodes
        """
        
        with gb._driver.session(database=gb.database) as session:
            result = session.run(query, {
                "min_connections": min_connections,
                "max_nodes": max_nodes
            })
            
            records = list(result)
            
            if records:  # Neo4j has data
                logger.info(f"Building graph from Neo4j ({len(records)} papers)")
                
                # Build title lookup from PaperIndex if Neo4j titles are 'Unknown'
                title_lookup = {}
                if any(r["paper_title"] in (None, "Unknown", "") for r in records):
                    try:
                        idx = PaperIndex()
                        all_papers = idx.get_all_papers(limit=500)
                        title_lookup = {p.doc_id: (p.title, p.year) for p in all_papers}
                        logger.info(f"Built PaperIndex title lookup ({len(title_lookup)} entries)")
                    except Exception as e:
                        logger.warning(f"PaperIndex lookup failed: {e}")
                
                for record in records:
                    paper_id = record["paper_id"]
                    paper_title = record["paper_title"]
                    paper_year = record["paper_year"]
                    
                    # Enrich from PaperIndex if title is missing
                    if not paper_title or paper_title in ("Unknown", ""):
                        if paper_id in title_lookup:
                            paper_title, paper_year = title_lookup[paper_id]
                        else:
                            paper_title = paper_title or "Untitled"
                    
                    # Add paper node
                    nodes.append(Node(
                        id=paper_id,
                        label=f"{paper_title[:30]}..." if len(paper_title) > 30 else paper_title,
                        title=f"{paper_title} ({paper_year or 'N/A'})",
                        color="#4ECDC4",
                        size=20,
                        type="Paper"
                    ))
                    
                    # Add species nodes and edges
                    for species_name in record["species"]:
                        if not species_name:
                            continue
                        
                        species_id = f"species_{species_name}"
                        
                        if not any(n.id == species_id for n in nodes):
                            nodes.append(Node(
                                id=species_id,
                                label=species_name,
                                title=f"Species: {species_name}",
                                color="#FF6B6B",
                                size=15,
                                type="Species"
                            ))
                        
                        edges.append(Edge(
                            source=paper_id,
                            target=species_id,
                            label="mentions"
                        ))
                
                # Add species co-occurrence edges (species that appear in the same papers)
                cooccur_query = """
                MATCH (p:Paper)-[:MENTIONS]->(s1:Species),
                      (p)-[:MENTIONS]->(s2:Species)
                WHERE id(s1) < id(s2)
                WITH s1.scientific_name AS sp1, s2.scientific_name AS sp2,
                     COUNT(DISTINCT p) AS shared_papers
                WHERE shared_papers >= 1
                RETURN sp1, sp2, shared_papers
                ORDER BY shared_papers DESC
                LIMIT 100
                """
                with gb._driver.session(database=gb.database) as session2:
                    cooccur_result = session2.run(cooccur_query)
                    for cr in cooccur_result:
                        src_id = f"species_{cr['sp1']}"
                        tgt_id = f"species_{cr['sp2']}"
                        # Only add edge if both species nodes exist
                        if any(n.id == src_id for n in nodes) and any(n.id == tgt_id for n in nodes):
                            edges.append(Edge(
                                source=src_id,
                                target=tgt_id,
                                label=f"co-occurs ({cr['shared_papers']})",
                                color="#FFD93D",
                                width=1 + min(cr["shared_papers"], 5)
                            ))
                
                # --- Location nodes ---
                location_query = """
                MATCH (p:Paper)-[r:REFERENCES_LOCATION]->(l:Location)
                WHERE p.doc_id IN $paper_ids
                RETURN p.doc_id AS paper_id,
                       l.location_id AS loc_id,
                       l.name AS loc_name,
                       l.country AS country,
                       l.habitat_type AS habitat
                """
                paper_ids = [rec["paper_id"] for rec in records]
                with gb._driver.session(database=gb.database) as session3:
                    loc_result = session3.run(location_query, {"paper_ids": paper_ids})
                    for lr in loc_result:
                        loc_node_id = f"location_{lr['loc_id']}"
                        loc_label = lr["loc_name"] or lr["loc_id"]
                        
                        # Add location node if not already present
                        if not any(n.id == loc_node_id for n in nodes):
                            tooltip = f"Location: {loc_label}"
                            if lr.get("country"):
                                tooltip += f" ({lr['country']})"
                            if lr.get("habitat"):
                                tooltip += f" | {lr['habitat']}"
                            
                            nodes.append(Node(
                                id=loc_node_id,
                                label=loc_label,
                                title=tooltip,
                                color="#F7B731",
                                size=12,
                                type="Location"
                            ))
                        
                        # Edge: Paper → Location
                        edges.append(Edge(
                            source=lr["paper_id"],
                            target=loc_node_id,
                            label="located_in"
                        ))
                
                gb.close()
                logger.info(f"Built graph with {len(nodes)} nodes, {len(edges)} edges (Neo4j)")
                return nodes, edges
            else:
                gb.close()
                raise Exception("No entity data in Neo4j, falling back to PaperIndex")
                
    except Exception as e:
        logger.warning(f"Neo4j unavailable or no entities: {e}. Using PaperIndex fallback.")
    
    # Fallback: Build graph from PaperIndex (domain-based connections)
    try:
        from src.search.paper_index import PaperIndex
        from collections import defaultdict
        import json
        
        idx = PaperIndex()
        papers = idx.get_all_papers(limit=max_nodes)
        
        if not papers:
            logger.warning("No papers in PaperIndex")
            return nodes, edges
        
        logger.info(f"Building domain-based graph from {len(papers)} papers")
        
        # Group papers by domain
        domain_papers = defaultdict(list)
        for paper in papers:
            # Add paper node
            nodes.append(Node(
                id=paper.doc_id,
                label=f"{paper.title[:30]}..." if len(paper.title) > 30 else paper.title,
                title=f"{paper.title} ({paper.year or 'N/A'})",
                color="#4ECDC4",
                size=20,
                type="Paper"
            ))
            
            # Track domains
            if paper.domains:
                if isinstance(paper.domains, str):
                    domains_dict = json.loads(paper.domains)
                else:
                    domains_dict = paper.domains
                
                for domain, score in domains_dict.items():
                    if score > 0.1:  # Only significant domains
                        domain_papers[domain].append(paper.doc_id)
        
        # Add domain nodes and connect papers
        for domain, paper_ids in domain_papers.items():
            if len(paper_ids) < 2:  # Skip domains with only 1 paper
                continue
            
            domain_id = f"domain_{domain}"
            
            # Add domain node
            nodes.append(Node(
                id=domain_id,
                label=domain.replace('_', ' ').title(),
                title=f"Domain: {domain} ({len(paper_ids)} papers)",
                color="#95E1D3",
                size=15 + len(paper_ids),
                type="Domain"
            ))
            
            # Connect papers to domain
            for paper_id in paper_ids:
                edges.append(Edge(
                    source=paper_id,
                    target=domain_id,
                    label="classified_as"
                ))
        
        logger.info(f"Built graph with {len(nodes)} nodes, {len(edges)} edges (PaperIndex)")
        return nodes, edges
        
    except Exception as e:
        logger.error(f"Failed to build graph from PaperIndex: {e}")
        return nodes, edges


def _handle_node_click(selected_node):
    """Handle node click event."""
    node_id = selected_node
    
    if node_id.startswith("species_"):
        node_type = "Species"
        actual_id = node_id.replace("species_", "")
    elif node_id.startswith("location_"):
        node_type = "Location"
        actual_id = node_id.replace("location_", "")
    elif node_id.startswith("domain_"):
        node_type = "Domain"
        actual_id = node_id.replace("domain_", "")
    else:
        node_type = "Paper"
        actual_id = node_id
    
    st.session_state.selected_node = actual_id
    st.session_state.selected_node_type = node_type


def _render_node_details_panel():
    """Render node details and associated chunks below the graph."""
    node_id = st.session_state.selected_node
    node_type = st.session_state.selected_node_type
    
    if node_type == "Paper":
        _render_paper_details_below(node_id)
    elif node_type == "Species":
        _render_species_details_below(node_id)
    elif node_type == "Location":
        _render_location_details_below(node_id)
    elif node_type == "Domain":
        _render_domain_details_below(node_id)
    else:
        st.info(f"Selected: {node_id} ({node_type})")


def _render_paper_details_below(doc_id):
    """Render paper details and chunks below the graph."""
    try:
        gb = GraphBuilder()
        metadata = gb.get_paper_metadata(doc_id)
        chunks_data = gb.get_paper_chunks(doc_id)
        gb.close()
        
        if not metadata:
            metadata = {}
        
        # ── Enrich from PaperIndex if Neo4j metadata is missing ────
        title = metadata.get("title")
        if not title or title in ("Unknown", "Untitled", ""):
            try:
                idx = PaperIndex()
                paper = idx.get_paper(doc_id)
                if paper:
                    metadata["title"] = paper.title or title or "Untitled"
                    metadata["year"] = metadata.get("year") or paper.year
                    metadata["abstract"] = metadata.get("abstract") or paper.abstract
                    metadata["doi"] = metadata.get("doi") or paper.doi
                    if paper.authors and not metadata.get("authors"):
                        metadata["authors"] = paper.authors if isinstance(paper.authors, list) else [paper.authors]
                    if paper.keywords and not metadata.get("keywords"):
                        metadata["keywords"] = paper.keywords
                    logger.info(f"Enriched paper {doc_id} from PaperIndex: {metadata['title'][:50]}")
            except Exception as e:
                logger.warning(f"PaperIndex fallback failed: {e}")
        
        # Paper header
        st.markdown(f"## 📄 {metadata.get('title', 'Untitled')}")
        
        # Metadata row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Year", metadata.get("year") or "N/A")
        with mcol2:
            species_list = metadata.get("species", [])
            st.metric("Species", len(species_list))
        with mcol3:
            locations = metadata.get("locations", [])
            st.metric("Locations", len(locations))
        with mcol4:
            st.metric("Chunks", len(chunks_data) if chunks_data else 0)
        
        # Authors
        if metadata.get("authors"):
            authors = metadata["authors"]
            if isinstance(authors, list):
                st.markdown(f"**Authors:** {', '.join(str(a) for a in authors[:5])}")
            else:
                st.markdown(f"**Authors:** {authors}")
        
        # DOI
        if metadata.get("doi"):
            st.markdown(f"**DOI:** [{metadata['doi']}]({metadata['doi']})")
        
        # Abstract
        if metadata.get("abstract"):
            with st.expander("📜 Abstract", expanded=True):
                st.write(metadata["abstract"])
        
        # Species and locations in columns
        col_sp, col_loc = st.columns(2)
        with col_sp:
            if species_list:
                with st.expander(f"🦎 Species ({len(species_list)})", expanded=True):
                    for sp in species_list[:20]:
                        st.markdown(f"- *{sp}*")
        with col_loc:
            if locations:
                with st.expander(f"📍 Locations ({len(locations)})", expanded=True):
                    for loc in locations:
                        st.markdown(f"- {loc}")
        
        # Source chunks
        if chunks_data:
            st.markdown("---")
            st.markdown("### 📦 Source Text Chunks")
            st.caption(f"{len(chunks_data)} chunks from this paper — the text that generated graph relationships")
            _render_chunks_list(chunks_data, metadata)
        else:
            st.markdown("---")
            st.info("📦 No chunks available. Qdrant may be offline or chunks were not indexed for this paper.")
        
    except Exception as e:
        st.error(f"Failed to load paper details: {e}")
        logger.exception("Paper details error")


def _render_species_details_below(scientific_name):
    """Render species details and evidence chunks below the graph."""
    st.markdown(f"## 🦎 *{scientific_name}*")
    
    try:
        gb = GraphBuilder()
        
        # Get papers mentioning this species
        papers_query = """
        MATCH (p:Paper)-[:MENTIONS]->(s:Species {scientific_name: $name})
        RETURN p.doc_id AS doc_id, p.title AS title, p.year AS year
        ORDER BY p.year DESC
        """
        # Get co-occurring species
        cooccur_query = """
        MATCH (p:Paper)-[:MENTIONS]->(s:Species {scientific_name: $name}),
              (p)-[:MENTIONS]->(s2:Species)
        WHERE s <> s2
        WITH s2.scientific_name AS species, COUNT(DISTINCT p) AS shared
        RETURN species, shared
        ORDER BY shared DESC
        LIMIT 10
        """
        
        with gb._driver.session(database=gb.database) as session:
            papers = [dict(r) for r in session.run(papers_query, {"name": scientific_name})]
            cooccurring = [dict(r) for r in session.run(cooccur_query, {"name": scientific_name})]
        
        # Metrics
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Papers", len(papers))
        with mcol2:
            st.metric("Co-occurring Species", len(cooccurring))
        
        # Papers list — clickable
        if papers:
            with st.expander(f"📄 Papers mentioning *{scientific_name}* ({len(papers)})", expanded=True):
                for i, p in enumerate(papers):
                    pcol1, pcol2 = st.columns([5, 1])
                    with pcol1:
                        st.markdown(f"**{p['title']}** ({p.get('year', 'N/A')})")
                    with pcol2:
                        if st.button("📖", key=f"sp_paper_{i}_{p['doc_id']}", help="View paper details"):
                            st.session_state.selected_node = p["doc_id"]
                            st.session_state.selected_node_type = "Paper"
                            st.rerun()
        
        # Co-occurring species
        if cooccurring:
            with st.expander(f"🔗 Co-occurring species ({len(cooccurring)})", expanded=True):
                for c in cooccurring:
                    st.markdown(f"- *{c['species']}* (shared in {c['shared']} papers)")
        
        # Evidence chunks — text where species is mentioned
        st.markdown("---")
        st.markdown("### 📦 Evidence Chunks")
        st.caption(f"Text passages where *{scientific_name}* is mentioned")
        
        _render_species_evidence_chunks(gb, scientific_name, papers)
        gb.close()
        
    except Exception as e:
        st.error(f"Failed to load species details: {e}")
        logger.exception("Species details error")


def _render_domain_details_below(domain_name):
    """Render domain details below the graph."""
    st.markdown(f"## 🏷️ {domain_name.replace('_', ' ').title()}")
    
    try:
        from src.search.paper_index import PaperIndex
        idx = PaperIndex()
        papers = idx.get_all_papers(limit=500)
        
        import json
        matching = []
        for p in papers:
            if p.domains:
                domains = json.loads(p.domains) if isinstance(p.domains, str) else p.domains
                if domain_name in domains and domains[domain_name] > 0.1:
                    matching.append(p)
        
        st.metric("Papers", len(matching))
        
        # Clickable papers
        if matching:
            with st.expander(f"📄 Papers in this domain ({len(matching)})", expanded=True):
                for i, p in enumerate(matching[:20]):
                    pcol1, pcol2 = st.columns([5, 1])
                    with pcol1:
                        st.markdown(f"**{p.title}** ({p.year or 'N/A'})")
                    with pcol2:
                        if st.button("📖", key=f"dom_paper_{i}_{p.doc_id}", help="View paper details"):
                            st.session_state.selected_node = p.doc_id
                            st.session_state.selected_node_type = "Paper"
                            st.rerun()
        
        # Source chunks from first few papers
        if matching:
            st.markdown("---")
            st.markdown("### 📦 Source Chunks")
            st.caption(f"Text from papers classified under this domain")
            _render_domain_evidence_chunks(matching[:5])
        
    except Exception as e:
        st.error(f"Failed to load domain details: {e}")


def _render_location_details_below(location_id):
    """Render location details and linked papers below the graph."""
    st.markdown(f"## 📍 Location Details")
    
    try:
        gb = GraphBuilder()
        
        # Get location info and linked papers
        query = """
        MATCH (l:Location {location_id: $loc_id})
        OPTIONAL MATCH (p:Paper)-[:REFERENCES_LOCATION]->(l)
        RETURN l.name AS name, l.country AS country,
               l.region AS region, l.habitat_type AS habitat,
               l.latitude AS lat, l.longitude AS lon,
               collect(DISTINCT {title: p.title, year: p.year, doc_id: p.doc_id}) AS papers
        """
        
        with gb._driver.session(database=gb.database) as session:
            result = session.run(query, {"loc_id": location_id})
            record = result.single()
        
        gb.close()
        
        if not record:
            st.warning(f"Location not found: {location_id}")
            return
        
        loc_name = record['name'] or location_id
        st.markdown(f"### {loc_name}")
        
        # Metadata row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Country", record.get("country") or "N/A")
        with mcol2:
            st.metric("Region", record.get("region") or "N/A")
        with mcol3:
            st.metric("Habitat", record.get("habitat") or "N/A")
        with mcol4:
            papers = [p for p in record.get("papers", []) if p.get("doc_id")]
            st.metric("Papers", len(papers))
        
        # Coordinates
        lat, lon = record.get("lat"), record.get("lon")
        if lat is not None and lon is not None:
            st.markdown(f"**Coordinates:** {lat:.4f}°N, {lon:.4f}°E")
        
        # Clickable papers
        if papers:
            with st.expander(f"📄 Papers referencing this location ({len(papers)})", expanded=True):
                for i, p in enumerate(papers):
                    pcol1, pcol2 = st.columns([5, 1])
                    with pcol1:
                        st.markdown(f"**{p.get('title', 'Untitled')}** ({p.get('year', 'N/A')})")
                    with pcol2:
                        if st.button("📖", key=f"loc_paper_{i}_{p['doc_id']}", help="View paper details"):
                            st.session_state.selected_node = p["doc_id"]
                            st.session_state.selected_node_type = "Paper"
                            st.rerun()
        
        # Source chunks mentioning this location
        if papers:
            st.markdown("---")
            st.markdown("### 📦 Evidence Chunks")
            st.caption(f"Text passages mentioning *{loc_name}*")
            _render_location_evidence_chunks(papers, loc_name)
    
    except Exception as e:
        st.error(f"Failed to load location details: {e}")
        logger.exception("Location details error")


def _render_chunks_list(chunks_data, metadata):
    """Render a list of text chunks with entity highlighting."""
    try:
        from src.ui.components.entity_highlighter import (
            highlight_entities, 
            create_legend
        )
        has_highlighter = True
    except ImportError:
        has_highlighter = False
    
    if has_highlighter:
        st.markdown(create_legend(), unsafe_allow_html=True)
    
    global_entities = {
        "species": metadata.get("species", []) if metadata else [],
        "locations": metadata.get("locations", []) if metadata else [],
    }
    
    # Group by section
    sections = {}
    for chunk in chunks_data:
        section = chunk.get("section") or "General"
        if section not in sections:
            sections[section] = []
        sections[section].append(chunk)
    
    for section, section_chunks in sections.items():
        with st.expander(f"📑 {section} ({len(section_chunks)} chunks)", expanded=True):
            for i, chunk in enumerate(section_chunks):
                text = chunk.get("text", "")
                word_count = chunk.get("word_count", len(text.split()))
                page = chunk.get("page") or "N/A"
                
                st.markdown(f"**Chunk {i+1}** • {word_count} words • Page {page}")
                
                display_text = text[:800] + "..." if len(text) > 800 else text
                
                if has_highlighter:
                    highlighted = highlight_entities(display_text, global_entities)
                    st.markdown(
                        f'<div style="background:rgba(30,41,59,0.6);padding:0.75rem;border-radius:8px;'
                        f'line-height:1.6;font-size:0.85rem;color:#cbd5e1;border:1px solid rgba(148,163,184,0.15)">'
                        f'{highlighted}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.text(display_text)
                
                if i < len(section_chunks) - 1:
                    st.markdown("")


def _render_species_evidence_chunks(gb, scientific_name, papers):
    """Find and display text chunks that mention a specific species."""
    if not papers:
        st.info("No papers found for this species.")
        return
    
    evidence_count = 0
    
    for paper in papers[:5]:  # Limit to 5 papers for performance
        doc_id = paper["doc_id"]
        title = paper.get("title", "Unknown")
        
        chunks = gb.get_paper_chunks(doc_id)
        if not chunks:
            continue
        
        # Filter chunks that mention the species
        name_lower = scientific_name.lower()
        matching_chunks = [
            c for c in chunks
            if name_lower in c.get("text", "").lower()
        ]
        
        if matching_chunks:
            with st.expander(f"📄 {title} ({len(matching_chunks)} mentions)", expanded=evidence_count == 0):
                for i, chunk in enumerate(matching_chunks[:3]):
                    text = chunk.get("text", "")
                    # Highlight the species name
                    import re
                    highlighted = re.sub(
                        re.escape(scientific_name),
                        f'<mark style="background:#10b981;color:#0f172a;padding:0 3px;border-radius:3px">{scientific_name}</mark>',
                        text[:600],
                        flags=re.IGNORECASE
                    )
                    
                    st.markdown(
                        f'<div style="background:rgba(30,41,59,0.6);padding:0.75rem;border-radius:8px;'
                        f'line-height:1.6;font-size:0.85rem;color:#cbd5e1;border:1px solid rgba(148,163,184,0.15);'
                        f'margin-bottom:0.5rem">{highlighted}</div>',
                        unsafe_allow_html=True
                    )
                evidence_count += 1
    
    if evidence_count == 0:
        st.info("No text chunks found mentioning this species. Chunks may not be indexed in Qdrant.")


def _render_domain_evidence_chunks(matching_papers):
    """Show chunks from papers classified under a domain."""
    import re
    evidence_count = 0

    for paper in matching_papers:
        try:
            gb = GraphBuilder()
            chunks = gb.get_paper_chunks(paper.doc_id)
            gb.close()
        except Exception:
            continue

        if not chunks:
            continue

        title = paper.title or "Unknown"
        # Show first 3 chunks from each paper
        preview = chunks[:3]
        if preview:
            with st.expander(f"📄 {title} ({len(chunks)} chunks total)", expanded=evidence_count == 0):
                for i, chunk in enumerate(preview):
                    text = chunk.get("text", "")
                    display = text[:600] + "..." if len(text) > 600 else text
                    st.markdown(
                        f'<div style="background:rgba(30,41,59,0.6);padding:0.75rem;border-radius:8px;'
                        f'line-height:1.6;font-size:0.85rem;color:#cbd5e1;border:1px solid rgba(148,163,184,0.15);'
                        f'margin-bottom:0.5rem">{display}</div>',
                        unsafe_allow_html=True,
                    )
            evidence_count += 1

    if evidence_count == 0:
        st.info("No text chunks found. Chunks may not be indexed in Qdrant.")


def _render_location_evidence_chunks(papers, location_name):
    """Find and display chunks that mention a specific location."""
    import re
    evidence_count = 0
    name_lower = location_name.lower()

    for paper in papers[:5]:
        doc_id = paper.get("doc_id") if isinstance(paper, dict) else paper
        title = paper.get("title", "Unknown") if isinstance(paper, dict) else doc_id

        try:
            gb = GraphBuilder()
            chunks = gb.get_paper_chunks(doc_id)
            gb.close()
        except Exception:
            continue

        if not chunks:
            continue

        matching_chunks = [
            c for c in chunks
            if name_lower in c.get("text", "").lower()
        ]

        if matching_chunks:
            with st.expander(f"📄 {title} ({len(matching_chunks)} mentions)", expanded=evidence_count == 0):
                for chunk in matching_chunks[:3]:
                    text = chunk.get("text", "")
                    highlighted = re.sub(
                        re.escape(location_name),
                        f'<mark style="background:#f59e0b;color:#0f172a;padding:0 3px;border-radius:3px">{location_name}</mark>',
                        text[:600],
                        flags=re.IGNORECASE,
                    )
                    st.markdown(
                        f'<div style="background:rgba(30,41,59,0.6);padding:0.75rem;border-radius:8px;'
                        f'line-height:1.6;font-size:0.85rem;color:#cbd5e1;border:1px solid rgba(148,163,184,0.15);'
                        f'margin-bottom:0.5rem">{highlighted}</div>',
                        unsafe_allow_html=True,
                    )
            evidence_count += 1

    if evidence_count == 0:
        st.info("No text chunks found mentioning this location. Chunks may not be indexed in Qdrant.")



# ══════════════════════════════════════════════════════════════════
#  Agraph Tab Config Helper
# ══════════════════════════════════════════════════════════════════

def _tab_agraph_config(height=550):
    """Return a standard agraph Config for themed tabs."""
    return Config(
        width="100%",
        height=height,
        directed=False,
        physics={
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -40000,
                "centralGravity": 0.35,
                "springLength": 160,
                "springConstant": 0.04,
                "damping": 0.09,
                "avoidOverlap": 0.5,
            },
            "stabilization": {"iterations": 120},
        },
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#FF6B6B",
        collapsible=True,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False},
    )


def _handle_tab_selection(selected, prefix, default_type="Paper"):
    """Handle agraph node selection for specialized tabs.
    
    Returns (node_id, node_type) or (None, None) if nothing selected.
    """
    if not selected:
        return None, None
    
    node_id = selected
    # Determine type by prefix
    if node_id.startswith("species_"):
        return node_id.replace("species_", ""), "Species"
    elif node_id.startswith("location_"):
        return node_id.replace("location_", ""), "Location"
    elif node_id.startswith("domain_"):
        return node_id.replace("domain_", ""), "Domain"
    elif node_id.startswith("method_"):
        return node_id.replace("method_", ""), "Methodology"
    elif node_id.startswith("author_"):
        return node_id.replace("author_", ""), "Author"
    else:
        return node_id, default_type


# ══════════════════════════════════════════════════════════════════
#  Tab 1: Species Co-occurrence
# ══════════════════════════════════════════════════════════════════

def _render_tab_species(graph_available):
    """Species co-occurrence graph using agraph."""
    st.markdown("### 🔬 Species Co-occurrence Network")
    st.caption("Species that appear together in papers are connected. Click a node to see details.")

    if not graph_available:
        st.warning("⚠️ Neo4j not available — species graph requires entity data.")
        return

    max_sp = st.slider("Max species", 20, 200, 80, key="sp_max")

    try:
        gb = GraphBuilder()
        nodes, edges = [], []

        with gb._driver.session(database=gb.database) as session:
            # Top species by paper count
            sp_result = session.run("""
                MATCH (p:Paper)-[:MENTIONS]->(s:Species)
                WITH s, count(DISTINCT p) AS cnt
                ORDER BY cnt DESC LIMIT $limit
                RETURN s.scientific_name AS name, cnt
            """, limit=max_sp)
            species_list = [dict(r) for r in sp_result]

            for sp in species_list:
                nodes.append(Node(
                    id=f"species_{sp['name']}",
                    label=sp["name"],
                    title=f"{sp['name']}\n{sp['cnt']} papers",
                    color="#FF6B6B",
                    size=10 + min(sp["cnt"] * 2, 20),
                    type="Species",
                ))

            # Co-occurrences
            names = [sp["name"] for sp in species_list]
            cooccur = session.run("""
                MATCH (s1:Species)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(s2:Species)
                WHERE s1.scientific_name IN $names AND s2.scientific_name IN $names
                  AND id(s1) < id(s2)
                WITH s1.scientific_name AS sp1, s2.scientific_name AS sp2,
                     count(DISTINCT p) AS shared
                WHERE shared >= 1
                RETURN sp1, sp2, shared
                ORDER BY shared DESC LIMIT 300
            """, names=names)
            for r in cooccur:
                edges.append(Edge(
                    source=f"species_{r['sp1']}",
                    target=f"species_{r['sp2']}",
                    label=f"{r['shared']}",
                    color="#FFD93D",
                    width=1 + min(r["shared"], 5),
                ))
        gb.close()

        if not nodes:
            st.info("No species found in the graph.")
            return

        st.caption(f"📊 {len(nodes)} species · {len(edges)} co-occurrences")
        selected = agraph(nodes=nodes, edges=edges, config=_tab_agraph_config(),
                          key="species_agraph")

        node_id, node_type = _handle_tab_selection(selected, "species_", "Species")
        if node_id:
            st.markdown("---")
            _render_species_details_below(node_id)

    except Exception as e:
        st.error(f"Species graph error: {e}")
        logger.exception("Species tab error")


# ══════════════════════════════════════════════════════════════════
#  Tab 2: Domain Clusters
# ══════════════════════════════════════════════════════════════════

def _render_tab_domains(papers):
    """Domain cluster graph using agraph — papers grouped by classification."""
    import json

    st.markdown("### 🏷️ Domain Clusters")
    st.caption("Papers clustered by their scientific domain. Click a domain or paper to explore.")

    if not papers:
        st.info("No papers indexed yet.")
        return

    max_papers = st.slider("Max papers", 20, 200, 80, key="dom_max")
    min_score = st.slider("Min domain score", 0.05, 0.50, 0.15, 0.05, key="dom_score",
                          help="Only show domains above this classification score")

    nodes, edges = [], []
    domain_set = set()

    for p in papers[:max_papers]:
        # Paper node
        label = f"{p.title[:28]}..." if len(p.title) > 28 else p.title
        nodes.append(Node(
            id=p.doc_id,
            label=label,
            title=f"{p.title} ({p.year or 'N/A'})",
            color="#4ECDC4",
            size=16,
            type="Paper",
        ))

        # Domain edges
        if p.domains:
            domains = json.loads(p.domains) if isinstance(p.domains, str) else p.domains
            for domain, score in domains.items():
                if score >= min_score:
                    domain_id = f"domain_{domain}"
                    domain_set.add((domain_id, domain))
                    edges.append(Edge(
                        source=p.doc_id,
                        target=domain_id,
                        label="",
                    ))

    # Domain nodes
    for domain_id, domain_name in domain_set:
        connected = sum(1 for e in edges if getattr(e, 'to', None) == domain_id or e.source == domain_id)
        nodes.append(Node(
            id=domain_id,
            label=domain_name.replace("_", " ").title(),
            title=f"Domain: {domain_name}\n{connected} papers",
            color="#95E1D3",
            size=15 + min(connected, 20),
            type="Domain",
        ))

    if not nodes:
        st.info("No domain data available.")
        return

    st.caption(f"📊 {len(papers[:max_papers])} papers · {len(domain_set)} domains")
    selected = agraph(nodes=nodes, edges=edges, config=_tab_agraph_config(),
                      key="domains_agraph")

    node_id, node_type = _handle_tab_selection(selected, "domain_", "Paper")
    if node_id:
        st.markdown("---")
        if node_type == "Domain":
            _render_domain_details_below(node_id)
        else:
            _render_paper_details_below(node_id)


# ══════════════════════════════════════════════════════════════════
#  Tab 3: Paper Network
# ══════════════════════════════════════════════════════════════════

def _render_tab_papers(papers, graph_available):
    """Paper network — papers connected by shared species/locations."""
    st.markdown("### 📄 Paper Network")
    st.caption("Papers connected by shared species or locations. Click a paper to see details & chunks.")

    if not papers:
        st.info("No papers indexed yet.")
        return

    max_p = st.slider("Max papers", 20, 150, 60, key="pap_max")

    nodes, edges = [], []
    title_lookup = {p.doc_id: p for p in papers}

    if graph_available:
        try:
            gb = GraphBuilder()
            with gb._driver.session(database=gb.database) as session:
                # Get papers with their species lists
                result = session.run("""
                    MATCH (p:Paper)-[:MENTIONS]->(s:Species)
                    WITH p, collect(DISTINCT s.scientific_name) AS species
                    RETURN p.doc_id AS doc_id, p.title AS title, p.year AS year, species
                    ORDER BY size(species) DESC LIMIT $limit
                """, limit=max_p)
                records = [dict(r) for r in result]

            gb.close()

            if records:
                # Paper nodes
                for rec in records:
                    doc_id = rec["doc_id"]
                    title = rec["title"] or (title_lookup[doc_id].title if doc_id in title_lookup else "Untitled")
                    label = f"{title[:28]}..." if len(title) > 28 else title
                    nodes.append(Node(
                        id=doc_id,
                        label=label,
                        title=f"{title} ({rec.get('year') or 'N/A'})\n{len(rec['species'])} species",
                        color="#4ECDC4",
                        size=14 + min(len(rec["species"]), 15),
                        type="Paper",
                    ))

                # Shared species edges
                for i, r1 in enumerate(records):
                    s1 = set(r1["species"])
                    for j in range(i + 1, len(records)):
                        s2 = set(records[j]["species"])
                        shared = s1 & s2
                        if shared:
                            edges.append(Edge(
                                source=r1["doc_id"],
                                target=records[j]["doc_id"],
                                label=f"{len(shared)} shared",
                                color="#FFD93D",
                                width=1 + min(len(shared), 5),
                            ))
        except Exception as e:
            logger.warning(f"Neo4j paper network error: {e}")

    # PaperIndex fallback
    if not nodes:
        for p in papers[:max_p]:
            label = f"{p.title[:28]}..." if len(p.title) > 28 else p.title
            nodes.append(Node(
                id=p.doc_id,
                label=label,
                title=f"{p.title} ({p.year or 'N/A'})",
                color="#4ECDC4",
                size=16,
                type="Paper",
            ))

    if not nodes:
        st.info("No papers available.")
        return

    st.caption(f"📊 {len(nodes)} papers · {len(edges)} connections")
    selected = agraph(nodes=nodes, edges=edges, config=_tab_agraph_config(),
                      key="papers_agraph")

    node_id, node_type = _handle_tab_selection(selected, "paper_", "Paper")
    if node_id:
        st.markdown("---")
        _render_paper_details_below(node_id)


# ══════════════════════════════════════════════════════════════════
#  Tab 4: Methodology (Study Type)
# ══════════════════════════════════════════════════════════════════

def _render_tab_methodology(papers):
    """Study type / methodology clusters."""
    st.markdown("### 🔧 Methodology Network")
    st.caption("Papers grouped by study type (field, laboratory, review, etc). Click to explore.")

    if not papers:
        st.info("No papers indexed yet.")
        return

    nodes, edges = [], []
    method_papers = {}  # method → [IndexedPaper]

    for p in papers:
        study_type = p.study_type or "unknown"
        method_papers.setdefault(study_type, []).append(p)

    # Method nodes
    for method, mpapers in method_papers.items():
        method_id = f"method_{method}"
        nodes.append(Node(
            id=method_id,
            label=method.replace("_", " ").title(),
            title=f"Methodology: {method}\n{len(mpapers)} papers",
            color="#F59E0B",
            size=18 + min(len(mpapers), 20),
            type="Methodology",
        ))

        # Paper nodes + edges
        for p in mpapers:
            label = f"{p.title[:28]}..." if len(p.title) > 28 else p.title
            if not any(n.id == p.doc_id for n in nodes):
                nodes.append(Node(
                    id=p.doc_id,
                    label=label,
                    title=f"{p.title} ({p.year or 'N/A'})",
                    color="#4ECDC4",
                    size=14,
                    type="Paper",
                ))
            edges.append(Edge(source=p.doc_id, target=method_id, label=""))

    if not nodes:
        st.info("No methodology data available.")
        return

    st.caption(f"📊 {len(papers)} papers · {len(method_papers)} study types")
    selected = agraph(nodes=nodes, edges=edges, config=_tab_agraph_config(),
                      key="method_agraph")

    node_id, node_type = _handle_tab_selection(selected, "method_", "Paper")
    if node_id:
        st.markdown("---")
        if node_type == "Methodology":
            _render_methodology_details_below(node_id, method_papers.get(node_id, []))
        else:
            _render_paper_details_below(node_id)


def _render_methodology_details_below(method_name, papers):
    """Render methodology detail panel."""
    st.markdown(f"## 🔧 {method_name.replace('_', ' ').title()}")
    st.metric("Papers", len(papers))

    if papers:
        with st.expander(f"📄 Papers ({len(papers)})", expanded=True):
            for i, p in enumerate(papers[:25]):
                pcol1, pcol2 = st.columns([5, 1])
                with pcol1:
                    st.markdown(f"**{p.title}** ({p.year or 'N/A'})")
                with pcol2:
                    if st.button("📖", key=f"meth_p_{i}_{p.doc_id}", help="View paper"):
                        st.session_state.selected_node = p.doc_id
                        st.session_state.selected_node_type = "Paper"
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  Tab 5: Authors
# ══════════════════════════════════════════════════════════════════

def _render_tab_authors(papers):
    """Author collaboration network using agraph."""
    import json

    st.markdown("### 👤 Author Collaboration Network")
    st.caption("Authors connected by co-authored papers. Click a node to see details.")

    if not papers:
        st.info("No papers indexed yet.")
        return

    max_authors = st.slider("Max authors", 20, 200, 80, key="auth_max")

    # Build author → papers mapping
    author_papers = {}  # author_name → [IndexedPaper]
    for p in papers:
        authors = p.authors
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except (json.JSONDecodeError, TypeError):
                authors = [authors] if authors else []
        if not authors:
            continue
        for a in authors:
            a_clean = a.strip()
            if a_clean and len(a_clean) > 2:
                author_papers.setdefault(a_clean, []).append(p)

    if not author_papers:
        st.info("No author data available.")
        return

    # Take top authors by paper count
    top_authors = sorted(author_papers.items(), key=lambda x: -len(x[1]))[:max_authors]
    top_names = {name for name, _ in top_authors}

    nodes, edges = [], []

    for name, apapers in top_authors:
        author_id = f"author_{name}"
        nodes.append(Node(
            id=author_id,
            label=name if len(name) <= 25 else name[:22] + "...",
            title=f"Author: {name}\n{len(apapers)} papers",
            color="#A78BFA",
            size=10 + min(len(apapers) * 3, 20),
            type="Author",
        ))

    # Collaboration edges (shared papers)
    top_list = list(top_authors)
    for i, (name1, papers1) in enumerate(top_list):
        ids1 = {p.doc_id for p in papers1}
        for j in range(i + 1, len(top_list)):
            name2, papers2 = top_list[j]
            ids2 = {p.doc_id for p in papers2}
            shared = ids1 & ids2
            if shared:
                edges.append(Edge(
                    source=f"author_{name1}",
                    target=f"author_{name2}",
                    label=f"{len(shared)}",
                    color="#C4B5FD",
                    width=1 + min(len(shared), 5),
                ))

    st.caption(f"📊 {len(nodes)} authors · {len(edges)} collaborations")
    selected = agraph(nodes=nodes, edges=edges, config=_tab_agraph_config(),
                      key="authors_agraph")

    node_id, node_type = _handle_tab_selection(selected, "author_", "Author")
    if node_id:
        st.markdown("---")
        _render_author_details_below(node_id, author_papers.get(node_id, []))


def _render_author_details_below(author_name, papers):
    """Render author detail panel."""
    st.markdown(f"## 👤 {author_name}")
    st.metric("Papers", len(papers))

    if papers:
        with st.expander(f"📄 Papers by {author_name} ({len(papers)})", expanded=True):
            for i, p in enumerate(papers[:25]):
                pcol1, pcol2 = st.columns([5, 1])
                with pcol1:
                    st.markdown(f"**{p.title}** ({p.year or 'N/A'})")
                with pcol2:
                    if st.button("📖", key=f"auth_p_{i}_{p.doc_id}", help="View paper"):
                        st.session_state.selected_node = p.doc_id
                        st.session_state.selected_node_type = "Paper"
                        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  Tab 6: Locations
# ══════════════════════════════════════════════════════════════════

def _render_tab_locations(graph_available):
    """Location network using agraph — locations connected to papers with year info."""
    st.markdown("### 📍 Location Network")
    st.caption("Geographic locations referenced in papers. Click to explore.")

    if not graph_available:
        st.warning("⚠️ Neo4j not available — location graph requires entity data.")
        return

    max_locs = st.slider("Max locations", 20, 200, 80, key="loc_max")

    try:
        gb = GraphBuilder()
        nodes, edges = [], []

        with gb._driver.session(database=gb.database) as session:
            result = session.run("""
                MATCH (p:Paper)-[:REFERENCES_LOCATION]->(l:Location)
                WITH l, collect(DISTINCT p) AS papers, count(DISTINCT p) AS cnt
                ORDER BY cnt DESC LIMIT $limit
                RETURN l.location_id AS loc_id, l.name AS name,
                       l.country AS country, l.habitat_type AS habitat, cnt,
                       [pp in papers | {doc_id: pp.doc_id, title: pp.title, year: pp.year}] AS papers
            """, limit=max_locs)
            records = [dict(r) for r in result]

        gb.close()

        if not records:
            st.info("No locations found in the graph.")
            return

        # Location nodes
        paper_ids_seen = set()
        for rec in records:
            loc_id = f"location_{rec['loc_id']}"
            loc_name = rec["name"] or rec["loc_id"]
            tooltip = f"Location: {loc_name}"
            if rec.get("country"):
                tooltip += f"\nCountry: {rec['country']}"
            if rec.get("habitat"):
                tooltip += f"\nHabitat: {rec['habitat']}"
            tooltip += f"\n{rec['cnt']} papers"

            nodes.append(Node(
                id=loc_id,
                label=loc_name if len(loc_name) <= 20 else loc_name[:17] + "...",
                title=tooltip,
                color="#F7B731",
                size=10 + min(rec["cnt"] * 2, 20),
                type="Location",
            ))

            # Connected paper nodes
            for p in rec["papers"][:10]:
                doc_id = p["doc_id"]
                if doc_id not in paper_ids_seen:
                    paper_ids_seen.add(doc_id)
                    title = p.get("title") or "Untitled"
                    year = p.get("year")
                    label = f"{title[:25]}..." if len(title) > 25 else title
                    nodes.append(Node(
                        id=doc_id,
                        label=label,
                        title=f"{title} ({year or 'N/A'})",
                        color="#4ECDC4",
                        size=14,
                        type="Paper",
                    ))
                edges.append(Edge(
                    source=doc_id,
                    target=loc_id,
                    label=str(p.get("year") or ""),
                ))

        st.caption(f"📊 {len(records)} locations · {len(paper_ids_seen)} papers")
        selected = agraph(nodes=nodes, edges=edges, config=_tab_agraph_config(),
                          key="locations_agraph")

        node_id, node_type = _handle_tab_selection(selected, "location_", "Paper")
        if node_id:
            st.markdown("---")
            if node_type == "Location":
                _render_location_details_below(node_id)
            else:
                _render_paper_details_below(node_id)

    except Exception as e:
        st.error(f"Location graph error: {e}")
        logger.exception("Locations tab error")



# ══════════════════════════════════════════════════════════════════
#  Pyvis-based Graph Views (ported from V1)
# ══════════════════════════════════════════════════════════════════

def _create_pyvis():
    """Create a configured Pyvis Network instance."""
    from pyvis.network import Network

    net = Network(
        notebook=False,
        bgcolor="#0f172a",
        font_color="#e2e8f0",
        height="580px",
        width="100%",
        cdn_resources="remote",
        select_menu=False,
        filter_menu=False,
    )
    net.force_atlas_2based(
        gravity=-80,
        central_gravity=0.005,
        spring_length=250,
        spring_strength=0.02,
        damping=0.9,
        overlap=0,
    )
    net.set_options("""
    {
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": false
        },
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 500,
                "fit": true
            },
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.005,
                "springLength": 250,
                "springConstant": 0.02,
                "damping": 0.9,
                "avoidOverlap": 0.5
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        }
    }
    """)
    return net


def _render_pyvis(net):
    """Render a Pyvis Network in Streamlit."""
    import streamlit.components.v1 as components

    html = net.generate_html()
    tooltip_css = """
    <style>
    body { margin: 0; overflow: hidden; }
    div.vis-tooltip {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-family: Inter, sans-serif !important;
        font-size: 12px !important;
        white-space: pre-line !important;
        max-width: 300px !important;
    }
    </style>
    """
    html = html.replace("</head>", tooltip_css + "</head>")
    components.html(html, height=600, scrolling=False)


# ── Species Graph ──────────────────────────────────────────────

def _render_species_pyvis(papers, graph_available):
    """Render species graph from Neo4j or inferred from keywords."""
    st.markdown("### Species Interaction Network")

    max_nodes = st.slider("Max species nodes", 10, 80, 40, key="species_max_nodes")

    if graph_available:
        st.caption("Species connections from the Neo4j knowledge graph")
        _species_from_neo4j(max_nodes)
    else:
        st.caption("Species co-occurrence inferred from paper keywords (Neo4j offline)")
        _species_from_keywords(papers, max_nodes)


def _species_from_neo4j(max_nodes):
    """Render species graph from Neo4j data."""
    try:
        from src.graph.network_analysis import detect_communities, community_color
        import networkx as nx

        gb = GraphBuilder()
        G = nx.Graph()

        with gb._driver.session(database=gb.database) as session:
            result = session.run("""
                MATCH (p:Paper)-[:MENTIONS]->(s:Species)
                WITH s, collect(p) as papers, count(p) as paper_count
                ORDER BY paper_count DESC LIMIT $limit
                RETURN s.scientific_name as name, paper_count,
                       [p in papers | p.title][..3] as paper_titles
            """, limit=max_nodes)

            for record in result:
                G.add_node(record["name"], label=record["name"],
                           count=record["paper_count"],
                           papers=record["paper_titles"],
                           node_type="species")

            result2 = session.run("""
                MATCH (s1:Species)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(s2:Species)
                WHERE id(s1) < id(s2)
                WITH s1.scientific_name AS sp1, s2.scientific_name AS sp2,
                     COUNT(DISTINCT p) AS shared
                WHERE shared >= 1
                RETURN sp1, sp2, shared
                ORDER BY shared DESC LIMIT 100
            """)
            for record in result2:
                if record["sp1"] in G and record["sp2"] in G:
                    G.add_edge(record["sp1"], record["sp2"],
                               weight=record["shared"],
                               title=f"{record['shared']} shared papers")

        gb.close()

        if G.number_of_nodes() == 0:
            st.info("No species found in the graph database.")
            return

        communities = detect_communities(G)
        net = _create_pyvis()

        for node_id, data in G.nodes(data=True):
            comm = communities.get(node_id, 0)
            size = 8 + data.get("count", 1) * 3
            papers_str = "\n".join(data.get("papers", [])[:3])
            net.add_node(
                str(node_id), label=data.get("label", str(node_id)),
                title=f"{node_id}\n{data.get('count', 0)} papers\n{papers_str}",
                size=size, color=community_color(comm),
                font={"size": 10, "color": "#e2e8f0"},
            )

        for u, v, data in G.edges(data=True):
            net.add_edge(str(u), str(v), title=data.get("title", ""),
                         color={"color": "#f59e0b80"}, width=1.5)

        _render_pyvis(net)
        st.caption(f"📊 {G.number_of_nodes()} species · {G.number_of_edges()} co-occurrences")

    except Exception as e:
        st.warning(f"Neo4j query error: {e}")
        logger.exception("Species graph Neo4j error")


def _species_from_keywords(papers, max_nodes):
    """Infer species from keywords and build co-occurrence graph."""
    from src.graph.network_analysis import detect_communities, community_color
    import networkx as nx

    species_papers = {}
    for p in papers:
        for kw in p.keywords:
            if len(kw.split()) >= 2 and kw[0].isupper():
                species_papers.setdefault(kw, []).append(p.doc_id)

    if not species_papers:
        st.info("No species-like keywords found.")
        return

    G = nx.Graph()
    sorted_species = sorted(species_papers.items(), key=lambda x: -len(x[1]))[:max_nodes]

    for sp_name, doc_ids in sorted_species:
        G.add_node(sp_name, label=sp_name, count=len(doc_ids), node_type="species")

    for i, (name1, docs1) in enumerate(sorted_species):
        for j in range(i + 1, len(sorted_species)):
            name2, docs2 = sorted_species[j]
            shared = set(docs1) & set(docs2)
            if shared:
                G.add_edge(name1, name2, weight=len(shared),
                           title=f"{len(shared)} shared papers")

    communities = detect_communities(G)
    net = _create_pyvis()

    for node_id, data in G.nodes(data=True):
        comm = communities.get(node_id, 0)
        size = 8 + data.get("count", 1) * 2
        net.add_node(
            str(node_id), label=data.get("label", str(node_id)),
            title=f"{node_id}\n{data.get('count', 0)} papers",
            size=size, color=community_color(comm),
            font={"size": 10, "color": "#e2e8f0"},
        )

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        net.add_edge(str(u), str(v), width=max(0.5, w * 1.5),
                     title=data.get("title", ""),
                     color={"color": "#ffffff25"})

    _render_pyvis(net)
    st.caption(f"📊 {G.number_of_nodes()} species · {G.number_of_edges()} co-occurrences")


# ── Paper Connections ──────────────────────────────────────────

def _render_papers_pyvis(papers):
    """Render paper similarity network with community coloring."""
    st.markdown("### Paper Similarity Network")
    st.caption("Papers connected by shared keywords/domains. Color = community, size = centrality.")

    max_nodes = st.slider("Max papers", 10, 80, 40, key="papers_max_nodes")

    from src.graph.network_analysis import (
        build_paper_graph, detect_communities, compute_centrality,
        community_color,
    )

    G = build_paper_graph(papers[:max_nodes], min_similarity=0.45)

    if G.number_of_nodes() == 0:
        st.info("Not enough papers to build similarity network.")
        return

    communities = detect_communities(G)
    centrality = compute_centrality(G)

    net = _create_pyvis()

    for node_id, data in G.nodes(data=True):
        comm = communities.get(node_id, 0)
        cent = centrality.get(node_id, 0)
        c = community_color(comm)
        size = 8 + cent * 40
        label = data.get("label", str(node_id)[:20])
        title = (f"{data.get('full_title', label)}\n"
                 f"{data.get('year', '')} · {data.get('domain', '')}\n"
                 f"Community {comm + 1}")

        net.add_node(
            str(node_id), label=label, title=title,
            size=size, color=c, font={"size": 9, "color": "#e2e8f0"},
        )

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.5)
        kws = data.get("shared_domains", [])
        title = ", ".join(d.replace('_', ' ') for d in kws[:3]) if kws else ""
        net.add_edge(str(u), str(v), width=max(0.5, w * 2), title=title,
                     color={"color": "#ffffff20", "opacity": 0.3})

    _render_pyvis(net)

    n_communities = len(set(communities.values())) if communities else 0
    st.caption(f"📊 {G.number_of_nodes()} papers · {G.number_of_edges()} connections · {n_communities} communities")


# ── Concept Map ────────────────────────────────────────────────

def _render_concepts_pyvis(papers):
    """Render keyword co-occurrence concept map."""
    import math

    st.markdown("### Concept Map (Domain & Keyword Co-occurrence)")
    st.caption(
        "Concepts that appear together across papers are connected. "
        "Size = paper count (log-scaled). Color = community cluster."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        max_nodes = st.slider("Max concepts", 10, 80, 40, key="concepts_max_nodes")
    with col_b:
        min_co = st.slider("Min co-occurrence", 2, 15, 5, key="concept_min_co",
                           help="Minimum papers where two concepts must co-occur")

    from src.graph.network_analysis import (
        build_concept_graph, detect_communities, compute_centrality,
        community_color,
    )

    G = build_concept_graph(papers, min_cooccurrence=min_co)

    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    if G.number_of_nodes() == 0:
        st.info("No concept co-occurrences found. Try lowering the minimum co-occurrence.")
        return

    communities = detect_communities(G)
    centrality = compute_centrality(G)

    net = _create_pyvis()

    for node_id, data in G.nodes(data=True):
        comm = communities.get(node_id, 0)
        c = community_color(comm)
        paper_count = data.get("paper_count", 1)
        size = min(25, 6 + math.log2(1 + paper_count) * 4)
        label = data.get("label", str(node_id))
        title = f"{label}\n{paper_count} papers · Community {comm + 1}"

        net.add_node(
            str(node_id), label=label, title=title,
            size=size, color=c, font={"size": 9, "color": "#e2e8f0"},
        )

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        net.add_edge(str(u), str(v), width=min(3, max(0.3, w * 0.5)),
                     title=data.get("title", ""),
                     color={"color": "#ffffff18"})

    _render_pyvis(net)

    n_communities = len(set(communities.values())) if communities else 0
    st.caption(f"📊 {G.number_of_nodes()} concepts · {G.number_of_edges()} links · {n_communities} clusters")


