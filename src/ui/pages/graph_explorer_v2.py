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
        '<div class="hero-subtitle">Click nodes to explore • Connected Papers style</div>',
        unsafe_allow_html=True,
    )
    
    # Initialize session state
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    if "selected_node_type" not in st.session_state:
        st.session_state.selected_node_type = None
    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = False
    
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
    
    # Layout: Main graph + Sidebar
    col_graph, col_sidebar = st.columns([2, 1])
    
    with col_graph:
        _render_graph_panel(graph_available)
    
    with col_sidebar:
        _render_sidebar_panel()


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
    
    # Controls row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        layout_mode = st.selectbox(
            "Layout",
            ["Force-Directed", "Hierarchical", "Circular"],
            index=0
        )
    
    with col2:
        min_connections = st.slider(
            "Min Connections",
            min_value=0,
            max_value=10,
            value=0,  # Changed to 0 to load ALL papers by default
            help="Hide nodes with fewer connections"
        )
    
    with col3:
        max_nodes = st.slider(
            "Max Nodes",
            min_value=50,
            max_value=500,
            value=200,  # Increased to 200 for better initial view
            step=50,
            help="Limit graph size (lazy loading)"
        )
    
    with col4:
        domain_filter = st.multiselect(
            "Domains",
            ["All", "ecology", "marine_biology", "climate_science", "remote_sensing"],
            default=["All"]
        )
    
    with col5:
        show_labels = st.checkbox("Show Labels", value=True)
    
    st.markdown("---")
    
    # Build and render graph
    try:
        nodes, edges = _build_graph_data(
            min_connections=min_connections,
            domain_filter=domain_filter if "All" not in domain_filter else None,
            max_nodes=max_nodes  # Lazy loading control
        )
        
        if not nodes:
            st.warning("No nodes to display with current filters.")
            return
        
        # Configure agraph
        config = Config(
            width="100%",
            height=600,
            directed=False,
            physics={
                "enabled": layout_mode == "Force-Directed",
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "centralGravity": 0.3,
                    "springLength": 200,
                    "springConstant": 0.04,
                    "damping": 0.09
                }
            },
            hierarchical=layout_mode == "Hierarchical",
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
    # Extract node ID and type
    node_id = selected_node
    
    # Determine node type from ID prefix
    if node_id.startswith("species_"):
        node_type = "Species"
        actual_id = node_id.replace("species_", "")
    elif node_id.startswith("location_"):
        node_type = "Location"
        actual_id = node_id.replace("location_", "")
    else:
        node_type = "Paper"
        actual_id = node_id
    
    st.session_state.selected_node = actual_id
    st.session_state.selected_node_type = node_type
    st.session_state.show_chunks = False


def _render_sidebar_panel():
    """Render the sidebar with node details."""
    st.markdown("### 📋 Node Details")
    
    if not st.session_state.selected_node:
        st.info("👈 Click a node in the graph to see details")
        return
    
    node_id = st.session_state.selected_node
    node_type = st.session_state.selected_node_type
    
    # Render details based on node type
    if node_type == "Paper":
        _render_paper_details(node_id)
    elif node_type == "Species":
        _render_species_details(node_id)
    elif node_type == "Location":
        _render_location_details(node_id)


def _render_paper_details(doc_id):
    """Render paper node details in sidebar."""
    try:
        gb = GraphBuilder()
        metadata = gb.get_paper_metadata(doc_id)
        gb.close()
        
        if not metadata:
            st.warning("Paper metadata not found")
            return
        
        # Header
        st.markdown(f"#### 📄 {metadata['title']}")
        
        # Metadata
        col1, col2 = st.columns(2)
        with col1:
            if metadata.get('year'):
                st.metric("Year", metadata['year'])
        with col2:
            if metadata.get('species'):
                st.metric("Species", len(metadata['species']))
        
        # Authors
        if metadata.get('authors'):
            st.markdown("**Authors:**")
            st.caption(", ".join(metadata['authors'][:5]))
        
        # Abstract
        if metadata.get('abstract'):
            with st.expander("📜 Abstract", expanded=True):
                st.write(metadata['abstract'])
        
        # Species mentioned
        if metadata.get('species'):
            with st.expander(f"🦎 Species ({len(metadata['species'])})", expanded=False):
                for species in metadata['species'][:10]:
                    st.markdown(f"- *{species}*")
        
        # Locations
        if metadata.get('locations'):
            with st.expander(f"📍 Locations ({len(metadata['locations'])})", expanded=False):
                for location in metadata['locations']:
                    st.markdown(f"- {location}")
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📦 View Chunks"):
                st.session_state.show_chunks = True
                st.rerun()
        
        with col2:
            if metadata.get('doi'):
                st.markdown(f"[🔗 DOI]({metadata['doi']})")
        
        # Show chunks if requested
        if st.session_state.show_chunks:
            _render_chunk_viewer(doc_id)
            
    except Exception as e:
        st.error(f"Failed to load paper details: {e}")
        logger.exception("Paper details error")


def _render_species_details(scientific_name):
    """Render species node details in sidebar."""
    st.markdown(f"#### 🦎 *{scientific_name}*")
    
    try:
        gb = GraphBuilder()
        papers = gb.get_species_papers(scientific_name)
        gb.close()
        
        st.metric("Papers", len(papers))
        
        if papers:
            st.markdown("**Mentioned in:**")
            for paper in papers[:10]:
                st.markdown(f"- {paper['title']} ({paper.get('year', 'N/A')})")
        
        if st.button("🔍 Validate with GBIF"):
            st.info("GBIF validation feature coming soon!")
            
    except Exception as e:
        st.error(f"Failed to load species details: {e}")


def _render_location_details(location_name):
    """Render location node details in sidebar."""
    st.markdown(f"#### 📍 {location_name}")
    
    st.info("Location details: studies, coordinates, habitat type (coming soon)")


def _render_chunk_viewer(doc_id):
    """Render chunk viewer for a paper with entity highlighting."""
    st.markdown("---")
    st.markdown("### 📦 Document Chunks")
    
    try:
        from src.ui.components.entity_highlighter import (
            highlight_entities, 
            extract_entities_from_chunk,
            create_legend
        )
        
        # Get chunks from GraphBuilder (uses Qdrant now)
        gb = GraphBuilder()
        chunks_data = gb.get_paper_chunks(doc_id)
        
        # Also get paper metadata for entity extraction
        metadata = gb.get_paper_metadata(doc_id)
        gb.close()
        
        if not chunks_data:
            st.info("No chunks found for this paper")
            return
        
        # Show legend
        st.markdown(create_legend(), unsafe_allow_html=True)
        
        # Group by section
        sections = {}
        for chunk in chunks_data:
            section = chunk.get("section") or "Unknown"
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk)
        
        # Prepare entities for highlighting (from metadata)
        global_entities = {
            "species": metadata.get("species", []) if metadata else [],
            "locations": metadata.get("locations", []) if metadata else [],
        }
        
        # Display chunks by section
        for section, section_chunks in sections.items():
            with st.expander(f"📑 {section} ({len(section_chunks)} chunks)", expanded=False):
                for i, chunk in enumerate(section_chunks):
                    # Metadata row
                    st.markdown(
                        f"**Chunk {i+1}** • {chunk.get('word_count', 0)} words • "
                        f"Page {chunk.get('page') or 'N/A'}"
                    )
                    
                    # Highlight entities in text
                    text = chunk.get("text", "")
                    if len(text) > 500:
                        text = text[:500] + "..."
                    
                    highlighted_text = highlight_entities(text, global_entities)
                    
                    # Render with HTML
                    st.markdown(
                        f'<div style="background:#f8f9fa;padding:0.75rem;border-radius:4px;'
                        f'line-height:1.6;font-size:0.9rem">{highlighted_text}</div>',
                        unsafe_allow_html=True
                    )
                    
                    if i < len(section_chunks) - 1:  # Divider except for last chunk
                        st.markdown("---")
                    
    except Exception as e:
        st.error(f"Failed to load chunks: {e}")
        logger.exception("Chunk viewer error")
