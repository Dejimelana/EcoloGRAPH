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
    
    if not graph_available:
        st.warning("⚠️ Neo4j not available. Some features will be limited.")
        return
    
    # Full-width graph
    _render_graph_panel(graph_available)
    
    # Node details + chunks BELOW the graph (scrollable)
    if st.session_state.selected_node:
        st.markdown("---")
        _render_node_details_panel()


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
                
                for record in records:
                    paper_id = record["paper_id"]
                    paper_title = record["paper_title"] or "Unknown"
                    paper_year = record["paper_year"]
                    
                    # Add paper node
                    nodes.append(Node(
                        id=paper_id,
                        label=f"{paper_title[:30]}..." if len(paper_title) > 30 else paper_title,
                        title=f"{paper_title} ({paper_year})",
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
            st.warning(f"Paper metadata not found for: {doc_id}")
            return
        
        # Paper header
        st.markdown(f"## 📄 {metadata.get('title', 'Untitled')}")
        
        # Metadata row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Year", metadata.get("year", "N/A"))
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
            st.markdown(f"**Authors:** {', '.join(metadata['authors'][:5])}")
        
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
        
        # Papers list
        if papers:
            with st.expander(f"📄 Papers mentioning *{scientific_name}* ({len(papers)})", expanded=True):
                for p in papers:
                    st.markdown(f"- **{p['title']}** ({p.get('year', 'N/A')})")
        
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
        
        if matching:
            with st.expander(f"📄 Papers in this domain ({len(matching)})", expanded=True):
                for p in matching[:20]:
                    st.markdown(f"- **{p.title}** ({p.year or 'N/A'})")
        
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
        
        st.markdown(f"### {record['name'] or location_id}")
        
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
        
        # Linked papers
        if papers:
            with st.expander(f"📄 Papers referencing this location ({len(papers)})", expanded=True):
                for p in papers:
                    st.markdown(f"- **{p.get('title', 'Untitled')}** ({p.get('year', 'N/A')})")
    
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

