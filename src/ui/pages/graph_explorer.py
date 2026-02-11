"""
EcoloGRAPH — Graph Explorer Page.

Interactive knowledge graph visualization using Pyvis + NetworkX.
Shows paper-species-domain connections with community detection.
"""
import json
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from src.ui.theme import inject_css


def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">🕸️ Knowledge Graph</div>'
        '<div class="hero-subtitle">Explore connections between papers, species, domains, and concepts</div>',
        unsafe_allow_html=True,
    )
    
    # Link to Graph Explorer V2 - DEPRECATED: Now integrated below
    # st.info(
    #     "💡 **Try the new Interactive Graph Explorer V2!** "
    #     "Navigate to **🔗 Graph V2** in the sidebar for a Connected Papers-style interface "
    #     "with click-to-explore nodes, metadata sidebar, and chunk viewer."
    # )

    # --- Check data sources ---
    papers, graph_available = _load_data()

    if not papers:
        st.info(
            "No papers indexed yet. Run the ingestion pipeline first:\n\n"
            "```\npython scripts/ingest.py data/raw/\n```"
        )
        return

    # --- Controls ---
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        view_mode = st.radio(
            "Visualization",
            ["📊 Domain Network", "📄 Paper Connections", "🔗 Concept Map", "🔬 Species Graph", "🎯 Interactive Explorer"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with col2:
        max_nodes = st.slider("Max nodes", 10, 80, 40)

    with col3:
        if graph_available:
            st.markdown("🟢 **Neo4j connected**")
        else:
            st.markdown("🟡 **SQLite only** (Neo4j offline)")

    st.markdown("---")

    # --- Render selected view ---
    if view_mode == "📊 Domain Network":
        _render_domain_network(papers, max_nodes)
    elif view_mode == "📄 Paper Connections":
        _render_paper_connections(papers, max_nodes)
    elif view_mode == "🔗 Concept Map":
        _render_concept_map(papers, max_nodes)
    elif view_mode == "🔬 Species Graph":
        _render_species_graph(papers, graph_available, max_nodes)
    elif view_mode == "🎯 Interactive Explorer":
        _render_interactive_explorer(graph_available)


def _load_data():
    """Load papers and check Neo4j availability."""
    papers = []
    graph_available = False

    try:
        from src.search.paper_index import PaperIndex
        idx = PaperIndex()
        papers = idx.get_all_papers(limit=200)
    except Exception:
        pass

    try:
        from src.graph.graph_builder import GraphBuilder
        gb = GraphBuilder()
        gb.close()
        graph_available = True
    except Exception:
        pass

    return papers, graph_available


# ── Domain Network ──────────────────────────────────────────────

def _render_domain_network(papers, max_nodes):
    """Render domain-to-paper network graph using Pyvis."""
    st.markdown("### Domain → Paper Network")
    st.caption("Each domain node connects to papers classified in that domain. Node size = paper count.")

    from src.graph.network_analysis import (
        build_domain_graph, detect_communities, compute_centrality,
        domain_color, community_color,
    )

    G = build_domain_graph(papers[:max_nodes])

    if G.number_of_nodes() == 0:
        st.info("No domain data available.")
        return

    communities = detect_communities(G)
    centrality = compute_centrality(G)

    # Build Pyvis
    net = _create_pyvis()

    for node_id, data in G.nodes(data=True):
        is_domain = data.get("node_type") == "domain"
        c = domain_color(data.get("domain", ""))
        size = 25 + data.get("size", 0) * 4 if is_domain else 8 + centrality.get(node_id, 0) * 30
        label = data.get("label", str(node_id))
        title = label if is_domain else f"{data.get('full_title', label)}\n{data.get('year', '')}"
        shape = "dot"

        net.add_node(
            str(node_id), label=label if is_domain else "",
            title=title, size=size, color=c,
            shape=shape, font={"size": 12 if is_domain else 8, "color": "#e2e8f0"},
        )

    for u, v, data in G.edges(data=True):
        net.add_edge(str(u), str(v), width=1, color={"color": "#ffffff15"})

    _render_pyvis(net)

    # Stats
    from collections import Counter
    domain_counts = Counter(p.primary_domain for p in papers if p.primary_domain)
    top = domain_counts.most_common(6)
    if top:
        st.markdown("---")
        cols = st.columns(len(top))
        for col, (domain, count) in zip(cols, top):
            col.metric(domain.replace("_", " ").title(), f"{count} papers")


# ── Paper Connections ───────────────────────────────────────────

def _render_paper_connections(papers, max_nodes):
    """Render paper similarity network with community coloring."""
    st.markdown("### Paper Similarity Network")
    st.caption("Papers connected by shared keywords/domains. Color = community, size = centrality.")

    from src.graph.network_analysis import (
        build_paper_graph, detect_communities, compute_centrality,
        community_color, domain_color,
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
        title = f"{data.get('full_title', label)}\n{data.get('year', '')} · {data.get('domain', '')}\nCommunity {comm + 1}"

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


# ── Concept Map ─────────────────────────────────────────────────

def _render_concept_map(papers, max_nodes):
    """Render keyword co-occurrence concept map."""
    import math

    st.markdown("### Concept Map (Domain & Keyword Co-occurrence)")
    st.caption(
        "Concepts that appear together across papers are connected. "
        "Size = paper count (log-scaled). Color = community cluster."
    )

    from src.graph.network_analysis import (
        build_concept_graph, detect_communities, compute_centrality,
        community_color,
    )

    min_co = st.slider("Min co-occurrence", 2, 15, 5, key="concept_min_co",
                        help="Minimum number of papers where two concepts must co-occur")

    G = build_concept_graph(papers, min_cooccurrence=min_co)

    # Limit to top N nodes by degree to keep it readable
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
        # Log scaling + cap to prevent giant nodes
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


# ── Species Graph ───────────────────────────────────────────────

def _render_species_graph(papers, graph_available, max_nodes):
    """Render species network from Neo4j or inferred from papers."""
    st.markdown("### Species Interaction Network")

    if graph_available:
        st.caption("Species connections from the Neo4j knowledge graph")
        _render_species_from_neo4j(max_nodes)
    else:
        st.caption("Species co-occurrence inferred from paper keywords (Neo4j offline)")
        _render_species_from_keywords(papers, max_nodes)


def _render_species_from_neo4j(max_nodes):
    """Render species graph from Neo4j data."""
    try:
        from src.graph.graph_builder import GraphBuilder
        from src.graph.network_analysis import detect_communities, community_color

        gb = GraphBuilder()
        import networkx as nx
        G = nx.Graph()

        with gb._driver.session(database=gb.database) as session:
            result = session.run("""
                MATCH (s:Species)-[:MENTIONED_IN]->(p:Paper)
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
                MATCH (s1:Species)-[r]->(s2:Species)
                WHERE type(r) <> 'MENTIONED_IN'
                RETURN s1.scientific_name as from_sp, s2.scientific_name as to_sp,
                       type(r) as rel_type LIMIT 100
            """)
            for record in result2:
                if record["from_sp"] in G and record["to_sp"] in G:
                    G.add_edge(record["from_sp"], record["to_sp"],
                               title=record["rel_type"].replace("_", " ").lower())

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

    except Exception as e:
        st.warning(f"Neo4j query error: {e}")


def _render_species_from_keywords(papers, max_nodes):
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


# ── Pyvis Renderer ──────────────────────────────────────────────

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
        gravity=-80,           # Increased repulsion between nodes
        central_gravity=0.005, # Reduced central pull
        spring_length=250,     # INCREASED: longer edges = more readable
        spring_strength=0.02,  # REDUCED: weaker springs = less oscillation
        damping=0.9,           # INCREASED: more damping = faster stabilization
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
    import tempfile
    import os

    # Generate HTML
    html = net.generate_html()

    # Inject dark-mode tooltip styling
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


# ── Interactive Explorer (Graph V2 Integration) ───────────────────────────────────

def _render_interactive_explorer(graph_available):
    """Render Interactive Explorer (Graph V2's domain-based visualization integrated)."""
    st.markdown("### 🎯 Interactive Explorer")
    st.caption("Click on nodes to explore · Domain-based paper connections")
    
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
        from src.search.paper_index import PaperIndex
        from collections import defaultdict
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layout = st.selectbox("Layout", ["Force-Directed", "Hierarchical"], index=0)
        with col2:
            max_nodes = st.slider("Max Papers", 20, 200, 100, 20)
        with col3:
            show_labels = st.checkbox("Show Labels", value=True)
        
        # Build graph
        nodes, edges = [], []
        idx = PaperIndex()
        papers = idx.get_all_papers(limit=max_nodes)
        
        if not papers:
            st.warning("No papers found")
            return
        
        # Group by domain
        domain_papers = defaultdict(list)
        
        for paper in papers:
            nodes.append(Node(
                id=paper.doc_id,
                label=f"{paper.title[:30]}..." if len(paper.title) > 30 else paper.title,
                title=f"{paper.title} ({paper.year or 'N/A'})",
                color="#4ECDC4",
                size=20
            ))
            
            if paper.domains:
                domains_dict = json.loads(paper.domains) if isinstance(paper.domains, str) else paper.domains
                for domain, score in domains_dict.items():
                    if score > 0.1:
                        domain_papers[domain].append(paper.doc_id)
        
        # Add domains
        for domain, paper_ids in domain_papers.items():
            if len(paper_ids) >= 2:
                nodes.append(Node(
                    id=f"domain_{domain}",
                    label=domain.replace('_', ' ').title(),
                    title=f"Domain: {domain} ({len(paper_ids)} papers)",
                    color="#95E1D3",
                    size=15 + min(len(paper_ids), 20)
                ))
                for pid in paper_ids:
                    edges.append(Edge(source=pid, target=f"domain_{domain}"))
        
        if not nodes:
            st.warning("No graph data")
            return
        
        config = Config(
            width="100%", height=650, directed=False,
            physics={"enabled": layout == "Force-Directed"},
            hierarchical=layout == "Hierarchical",
            nodeHighlightBehavior=True,
            node={'labelProperty': 'label' if show_labels else None}
        )
        
        st.markdown(f"**📊 Graph:** {len(nodes)} nodes, {len(edges)} edges")
        selected = agraph(nodes=nodes, edges=edges, config=config)
        
        if selected:
            st.sidebar.markdown(f"### 🔍 Selected:\n`{selected}`")
        
    except ImportError as e:
        st.error(f"❌ Missing dependency: {e}\n\nInstall: `pip install streamlit-agraph`")
    except Exception as e:
        st.error(f"Error: {e}")
        import logging
        logging.getLogger(__name__).exception("Interactive explorer error")


#  ── Interactive Explorer (Graph V2 Integration) ───────────────────────────────────

def _render_interactive_explorer(graph_available):
    """Render Interactive Explorer with tabbed modes: Domain-based and Entity-based."""
    st.markdown("### 🎯 Interactive Explorer")
    st.caption("Explore paper connections interactively")
    
    # Initialize session state for node selection
    if "selected_node_id" not in st.session_state:
        st.session_state.selected_node_id = None
    if "selected_node_type" not in st.session_state:
        st.session_state.selected_node_type = None
    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = False
    
    # Tabs for different graph modes
    tab1, tab2 = st.tabs(["📊 Domain-based", "🔬 Entity-based"])
    
    with tab1:
        _render_domain_graph()
    
    with tab2:
        _render_entity_graph(graph_available)
    
    # Shared sidebar for node details
    if st.session_state.selected_node_id:
        with st.sidebar:
            _render_node_sidebar()


def _render_domain_graph():
    """Render domain-based graph (papers connected via shared domains)."""
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
        from src.search.paper_index import PaperIndex
        from collections import defaultdict
        import json
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layout = st.selectbox("Layout", ["Force-Directed", "Hierarchical"], index=0, key="domain_layout")
        with col2:
            max_nodes = st.slider("Max Papers", 20, 200, 100, 20, key="domain_max")
        with col3:
            show_labels = st.checkbox("Show Labels", value=True, key="domain_labels")
        
        # Build graph
        nodes, edges = [], []
        idx = PaperIndex()
        papers = idx.get_all_papers(limit=max_nodes)
        
        if not papers:
            st.warning("No papers found")
            return
        
        domain_papers = defaultdict(list)
        
        for paper in papers:
            nodes.append(Node(
                id=paper.doc_id,
                label=f"{paper.title[:30]}..." if len(paper.title) > 30 else paper.title,
                title=f"{paper.title} ({paper.year or 'N/A'})",
                color="#4ECDC4",
                size=20
            ))
            
            if paper.domains:
                domains_dict = json.loads(paper.domains) if isinstance(paper.domains, str) else paper.domains
                for domain, score in domains_dict.items():
                    if score > 0.1:
                        domain_papers[domain].append(paper.doc_id)
        
        for domain, paper_ids in domain_papers.items():
            if len(paper_ids) >= 2:
                nodes.append(Node(
                    id=f"domain_{domain}",
                    label=domain.replace('_', ' ').title(),
                    title=f"Domain: {domain} ({len(paper_ids)} papers)",
                    color="#95E1D3",
                    size=15 + min(len(paper_ids), 20)
                ))
                for pid in paper_ids:
                    edges.append(Edge(source=pid, target=f"domain_{domain}"))
        
        if not nodes:
            st.warning("No graph data")
            return
        
        config = Config(
            width="100%", height=600, directed=False,
            physics={"enabled": layout == "Force-Directed"},
            hierarchical=layout == "Hierarchical",
            nodeHighlightBehavior=True,
            node={'labelProperty': 'label' if show_labels else None}
        )
        
        st.markdown(f"**📊 Graph:** {len(nodes)} nodes, {len(edges)} edges")
        selected = agraph(nodes=nodes, edges=edges, config=config)
        
        # Handle node click
        if selected:
            if selected.startswith("domain_"):
                st.session_state.selected_node_type = "Domain"
                st.session_state.selected_node_id = selected.replace("domain_", "")
            else:
                st.session_state.selected_node_type = "Paper"
                st.session_state.selected_node_id = selected
            st.session_state.show_chunks = False
            st.rerun()
        
    except ImportError as e:
        st.error(f"Missing dependency: {e}\n\nInstall: pip install streamlit-agraph")
    except Exception as e:
        st.error(f"Error: {e}")
        import logging
        logging.getLogger(__name__).exception("Domain graph error")


def _render_entity_graph(graph_available):
    """Render entity-based graph (papers connected via Species/Locations from Neo4j)."""
    if not graph_available:
        st.info("🔬 **Entity-based graph requires Neo4j with extracted entities**\n\n"
                "Run ingestion WITH entity extraction:\n```\npython scripts/ingest.py data/raw/\n```")
        return
    
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
        from src.graph.graph_builder import GraphBuilder
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            layout = st.selectbox("Layout", ["Force-Directed", "Hierarchical"], index=0, key="entity_layout")
        with col2:
            min_connections = st.slider("Min Connections", 0, 10, 1, key="entity_min")
        with col3:
            max_nodes = st.slider("Max Papers", 20, 200, 100, 20, key="entity_max")
        
        # Build graph from Neo4j
        nodes, edges = [], []
        gb = GraphBuilder()
        
        # Fixed query: aggregate first, filter, then collect in subquery
        query = """
        MATCH (p:Paper)-[r:MENTIONS]->(s:Species)
        WITH p, count(r) as mention_count
        WHERE mention_count >= $min_connections
        WITH p, mention_count
        ORDER BY mention_count DESC
        LIMIT $max_nodes
        OPTIONAL MATCH (p)-[:MENTIONS]->(species:Species)
        OPTIONAL MATCH (p)-[:REFERENCES_LOCATION]->(location:Location)
        RETURN 
            p.doc_id as paper_id,
            p.title as paper_title,
            p.year as paper_year,
            collect(DISTINCT species.scientific_name) as species,
            collect(DISTINCT location.name) as locations
        """
        
        with gb._driver.session(database=gb.database) as session:
            result = session.run(query, {
                "min_connections": min_connections,
                "max_nodes": max_nodes
            })
            
            for record in result:
                paper_id = record["paper_id"]
                paper_title = record["paper_title"] or "Unknown"
                paper_year = record["paper_year"]
                
                # Add paper node
                nodes.append(Node(
                    id=paper_id,
                    label=f"{paper_title[:30]}..." if len(paper_title) > 30 else paper_title,
                    title=f"{paper_title} ({paper_year})",
                    color="#4ECDC4",
                    size=20
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
                            size=15
                        ))
                    
                    edges.append(Edge(source=paper_id, target=species_id, label="MENTIONS"))
                
                # Add location nodes and edges
                for location_name in record["locations"]:
                    if not location_name:
                        continue
                    
                    location_id = f"location_{location_name}"
                    
                    if not any(n.id == location_id for n in nodes):
                        nodes.append(Node(
                            id=location_id,
                            label=location_name,
                            title=f"Location: {location_name}",
                            color="#95E1D3",
                            size=15
                        ))
                    
                    edges.append(Edge(source=paper_id, target=location_id, label="STUDIED_AT"))
        
        gb.close()
        
        if not nodes:
            st.warning("No entity data found. Run ingestion with entity extraction enabled.")
            return
        
        config = Config(
            width="100%", height=600, directed=False,
            physics={"enabled": layout == "Force-Directed"},
            hierarchical=layout == "Hierarchical",
            nodeHighlightBehavior=True,
            node={'labelProperty': 'label'}
        )
        
        st.markdown(f"**🔬 Graph:** {len(nodes)} nodes, {len(edges)} edges")
        selected = agraph(nodes=nodes, edges=edges, config=config)
        
        # Handle node click
        if selected:
            if selected.startswith("species_"):
                st.session_state.selected_node_type = "Species"
                st.session_state.selected_node_id = selected.replace("species_", "")
            elif selected.startswith("location_"):
                st.session_state.selected_node_type = "Location"
                st.session_state.selected_node_id = selected.replace("location_", "")
            else:
                st.session_state.selected_node_type = "Paper"
                st.session_state.selected_node_id = selected
            st.session_state.show_chunks = False
            st.rerun()
        
    except Exception as e:
        st.error(f"Error loading entity graph: {e}")
        import logging
        logging.getLogger(__name__).exception("Entity graph error")


def _render_node_sidebar():
    """Render sidebar with details for selected node."""
    node_id = st.session_state.selected_node_id
    node_type = st.session_state.selected_node_type
    
    st.markdown(f"### 📋 {node_type} Details")
    st.markdown(f"**ID:** `{node_id}`")
    
    if node_type == "Paper":
        _render_paper_sidebar(node_id)
    elif node_type == "Species":
        st.markdown(f"**Species:** *{node_id}*")
        # Could add GBIF links here
    elif node_type == "Location":
        st.markdown(f"**Location:** {node_id}")
        # Could add map visualization here
    elif node_type == "Domain":
        st.markdown(f"**Domain:** {node_id.replace('_', ' ').title()}")


def _render_paper_sidebar(doc_id):
    """Render paper metadata and chunks in sidebar."""
    try:
        from src.search.paper_index import PaperIndex
        
        idx = PaperIndex()
        paper = idx.get_paper(doc_id)
        
        if not paper:
            st.warning("Paper not found")
            return
        
        # Title and year
        st.markdown(f"**{paper.title}**")
        if paper.year:
            st.caption(f"Year: {paper.year}")
        
        # Abstract
        if paper.abstract:
            with st.expander("📜 Abstract", expanded=True):
                st.write(paper.abstract[:500] + ("..." if len(paper.abstract) > 500 else ""))
        
        # Domains
        if paper.domains:
            import json
            domains_dict = json.loads(paper.domains) if isinstance(paper.domains, str) else paper.domains
            top_domains = sorted(domains_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            st.markdown("**Top Domains:**")
            for domain, score in top_domains:
                st.caption(f"• {domain.replace('_', ' ').title()} ({score:.2f})")
        
        # View Chunks button - made more prominent
        st.markdown("---")
        st.markdown("**📦 Document Chunks**")
        
        if st.session_state.show_chunks:
            if st.button("❌ Hide Chunks", key="hide_chunks_btn"):
                st.session_state.show_chunks = False
                st.rerun()
            _render_chunks(doc_id)
        else:
            if st.button("👁️ View Chunks", key="show_chunks_btn", type="primary"):
                st.session_state.show_chunks = True
                st.rerun()
        
    except Exception as e:
        st.error(f"Error loading paper: {e}")


def _render_chunks(doc_id):
    """Render paper chunks with entity highlighting."""
    try:
        from src.graph.graph_builder import GraphBuilder
        
        st.markdown("---")
        st.markdown("### 📦 Document Chunks")
        
        gb = GraphBuilder()
        chunks = gb.get_paper_chunks(doc_id)
        gb.close()
        
        if not chunks:
            st.info("No chunks found for this paper")
            return
        
        # Group by section
        sections = {}
        for chunk in chunks:
            section = chunk.get("section") or "Unknown"
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk)
        
        # Display chunks
        for section, section_chunks in sections.items():
            with st.expander(f"📑 {section} ({len(section_chunks)} chunks)", expanded=False):
                for i, chunk in enumerate(section_chunks):
                    st.markdown(f"**Chunk {i+1}** • {chunk.get('word_count', 0)} words")
                    text = chunk.get("text", "")
                    if len(text) > 300:
                        text = text[:300] + "..."
                    st.caption(text)
                    if i < len(section_chunks) - 1:
                        st.markdown("---")
        
    except Exception as e:
        st.error(f"Error loading chunks: {e}")
        import logging
        logging.getLogger(__name__).exception("Chunks error")
