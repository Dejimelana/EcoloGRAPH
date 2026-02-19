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

