"""
EcoloGRAPH — Dashboard Page.

Overview of the indexed knowledge base: papers, domains, species counts.
"""
import streamlit as st
from src.ui.theme import inject_css, metric_card


def render():
    inject_css()

    # Hero
    st.markdown(
        '<div class="hero-title">🌿 EcoloGRAPH</div>'
        '<div class="hero-subtitle">Graph RAG for Ecological Research — Knowledge Base Overview</div>',
        unsafe_allow_html=True,
    )

    # --- Service status ---
    col_status = st.columns(3)
    services = _check_services()
    for col, (name, online) in zip(col_status, services):
        dot = "🟢" if online else "🔴"
        col.markdown(f"{dot} **{name}**")

    st.markdown("---")

    # --- Metrics row ---
    paper_count, domain_count, domain_stats = _get_index_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card(paper_count, "Papers Indexed", "📄"), unsafe_allow_html=True)
    c2.markdown(metric_card(domain_count, "Active Domains", "🏷️"), unsafe_allow_html=True)
    c3.markdown(metric_card(8, "Agent Tools", "🔧"), unsafe_allow_html=True)
    c4.markdown(metric_card(43, "Scientific Domains", "🔬"), unsafe_allow_html=True)

    st.markdown("---")

    # --- Domain distribution ---
    if domain_stats:
        st.subheader("📊 Domain Distribution")
        st.caption("Click a domain on the Papers page to filter by it")
        import pandas as pd
        df = pd.DataFrame(
            [(d.replace("_", " ").title(), c) for d, c in sorted(domain_stats.items(), key=lambda x: -x[1])],
            columns=["Domain", "Papers"],
        )
        st.bar_chart(df.set_index("Domain"), color="#10b981", height=350)

        # Domain badges
        badges_html = '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.5rem">'
        for domain, count in sorted(domain_stats.items(), key=lambda x: -x[1]):
            badges_html += (
                f'<span class="domain-badge">'
                f'{domain.replace("_", " ").title()} · {count}'
                f'</span>'
            )
        badges_html += '</div>'
        st.markdown(badges_html, unsafe_allow_html=True)
    else:
        st.info(
            "No papers indexed yet. Run the ingestion pipeline first:\n\n"
            "```\npython scripts/ingest.py data/raw/\n```"
        )

    st.markdown("---")

    # --- Knowledge Graph Preview ---
    st.subheader("🕸️ Knowledge Graph Preview")
    st.caption("Interactive preview of the knowledge graph — click nodes to explore")
    _render_mini_graph()

    st.markdown("---")

    # --- Recent Papers ---
    if paper_count > 0:
        st.subheader("📄 Recent Papers")
        try:
            from src.search.paper_index import PaperIndex
            idx = PaperIndex()
            recent = idx.get_all_papers(limit=5)
            for i, p in enumerate(recent):
                domain_label = (p.primary_domain or "unknown").replace("_", " ").title()
                authors_str = ", ".join(p.authors[:2])
                if len(p.authors) > 2:
                    authors_str += f" +{len(p.authors) - 2}"

                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.markdown(
                        f'<div class="glass-card" style="padding:0.7rem 1rem;margin-bottom:0.4rem">'
                        f'<div style="display:flex;align-items:center;gap:0.6rem">'
                        f'<span class="domain-badge">{domain_label}</span>'
                        f'<span style="color:#64748b;font-size:0.8rem">{p.year or ""}</span>'
                        f'</div>'
                        f'<div style="color:#e2e8f0;font-weight:500;margin-top:0.3rem">{p.title or "Untitled"}</div>'
                        f'<div style="color:#94a3b8;font-size:0.8rem;margin-top:0.2rem">{authors_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("📖", key=f"dash_paper_{i}", help="View paper details"):
                        st.session_state.selected_paper_id = p.doc_id
                        st.session_state.nav_page = "📄 Papers"
                        st.rerun()
        except Exception:
            pass

    st.markdown("---")

    # --- Module overview ---
    st.subheader("🏗️ System Architecture")

    modules = [
        ("📥 Ingestion", "PDF parsing (Docling) + section-aware chunking"),
        ("🏷️ Extraction", "43-domain classifier + LLM entity extraction"),
        ("🔍 Search", "SQLite FTS5 (BM25) + Qdrant semantic + hybrid reranking"),
        ("🕸️ Graph", "Neo4j knowledge graph with 15 Cypher query templates"),
        ("🔗 Inference", "Cross-domain linker + hypothesis generator"),
        ("🤖 Agent", "LangGraph two-tier agent with 8 tools + multilingual"),
        ("🌐 Scrapers", "FishBase + GBIF + IUCN Red List API clients"),
    ]

    cols = st.columns(2)
    for i, (name, desc) in enumerate(modules):
        with cols[i % 2]:
            st.markdown(
                f'<div class="glass-card">'
                f'<div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.3rem">{name}</div>'
                f'<div style="font-size:0.85rem;color:#94a3b8">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )



def _render_mini_graph():
    """Render an interactive mini knowledge graph preview."""
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
        from src.graph.graph_builder import GraphBuilder
        import logging

        logger = logging.getLogger(__name__)

        gb = GraphBuilder()

        # Query top papers with most species mentions
        query = """
        MATCH (p:Paper)-[:MENTIONS]->(s:Species)
        WITH p, collect(DISTINCT s.scientific_name) AS species_list,
             count(DISTINCT s) AS sp_count
        ORDER BY sp_count DESC
        LIMIT 10
        RETURN p.doc_id AS paper_id, p.title AS title,
               p.year AS year, species_list
        """

        nodes = []
        edges = []
        seen_species = set()
        paper_ids = []

        with gb._driver.session(database=gb.database) as session:
            records = list(session.run(query))

            if not records:
                gb.close()
                st.info(
                    "No graph data yet. Ingest papers with entity extraction "
                    "to see the knowledge graph preview."
                )
                return

            for rec in records:
                pid = rec["paper_id"]
                paper_ids.append(pid)
                title = rec["title"] or "Untitled"

                nodes.append(Node(
                    id=pid,
                    label=f"{title[:25]}…" if len(title) > 25 else title,
                    title=f"{title} ({rec.get('year', 'N/A')})",
                    color="#4ECDC4",
                    size=18,
                    type="Paper",
                ))

                for sp in (rec["species_list"] or [])[:5]:
                    sp_id = f"species_{sp}"
                    if sp_id not in seen_species:
                        seen_species.add(sp_id)
                        nodes.append(Node(
                            id=sp_id,
                            label=sp,
                            title=f"Species: {sp}",
                            color="#FF6B6B",
                            size=13,
                            type="Species",
                        ))
                    edges.append(Edge(source=pid, target=sp_id, label="mentions"))

        # Location nodes
        if paper_ids:
            loc_query = """
            MATCH (p:Paper)-[:REFERENCES_LOCATION]->(l:Location)
            WHERE p.doc_id IN $paper_ids
            RETURN p.doc_id AS paper_id, l.location_id AS loc_id,
                   l.name AS loc_name, l.country AS country
            """
            seen_locs = set()
            with gb._driver.session(database=gb.database) as session2:
                for lr in session2.run(loc_query, {"paper_ids": paper_ids}):
                    loc_nid = f"location_{lr['loc_id']}"
                    loc_label = lr["loc_name"] or lr["loc_id"]
                    if loc_nid not in seen_locs:
                        seen_locs.add(loc_nid)
                        tooltip = f"Location: {loc_label}"
                        if lr.get("country"):
                            tooltip += f" ({lr['country']})"
                        nodes.append(Node(
                            id=loc_nid,
                            label=loc_label,
                            title=tooltip,
                            color="#F7B731",
                            size=11,
                            type="Location",
                        ))
                    edges.append(Edge(
                        source=lr["paper_id"],
                        target=loc_nid,
                        label="located_in",
                    ))

        gb.close()

        # Legend
        legend_html = (
            '<div style="display:flex;gap:1.2rem;margin-bottom:0.5rem;'
            'align-items:center;font-size:0.82rem;color:#94a3b8">'
            '<span>●&nbsp;<span style="color:#4ECDC4">Papers</span></span>'
            '<span>●&nbsp;<span style="color:#FF6B6B">Species</span></span>'
            '<span>●&nbsp;<span style="color:#F7B731">Locations</span></span>'
            f'<span style="margin-left:auto">{len(nodes)} nodes · '
            f'{len(edges)} edges</span>'
            '</div>'
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        # Render mini-graph
        config = Config(
            width="100%",
            height=350,
            directed=False,
            physics={
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -26000,
                    "centralGravity": 0.005,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.5,
                },
                "stabilization": {"iterations": 100},
            },
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#FF6B6B",
            collapsible=False,
            node={"labelProperty": "label"},
            link={"labelProperty": "label", "renderLabel": False},
        )

        agraph(nodes=nodes, edges=edges, config=config)

        # Navigation button
        if st.button("🕸️ Open Full Graph Explorer", use_container_width=True):
            st.session_state.nav_page = "🕸️ Graph Explorer"
            st.rerun()

    except Exception as e:
        st.info(
            "Knowledge graph preview unavailable. "
            "Start Neo4j and ingest papers to enable this feature."
        )
        logging.getLogger(__name__).debug(f"Mini-graph error: {e}")


def _check_services():
    """Check availability of backend services."""
    services = []

    # SQLite PaperIndex
    try:
        from src.search.paper_index import PaperIndex
        PaperIndex()
        services.append(("SQLite Index", True))
    except Exception:
        services.append(("SQLite Index", False))

    # Qdrant
    try:
        from src.retrieval.vector_store import VectorStore
        VectorStore()
        services.append(("Qdrant Vector DB", True))
    except Exception:
        services.append(("Qdrant Vector DB", False))

    # Neo4j
    try:
        from src.graph.graph_builder import GraphBuilder
        GraphBuilder()
        services.append(("Neo4j Graph DB", True))
    except Exception:
        services.append(("Neo4j Graph DB", False))

    return services


def _get_index_stats():
    """Get paper index statistics."""
    try:
        from src.search.paper_index import PaperIndex
        idx = PaperIndex()
        count = idx.count()

        # Get domain distribution using the actual API
        domain_stats = {}
        if count > 0:
            domain_list = idx.get_domains()  # returns [(domain, count), ...]
            domain_stats = {domain: cnt for domain, cnt in domain_list}

        active_domains = len(domain_stats)
        return count, active_domains, domain_stats
    except Exception:
        return 0, 0, {}

