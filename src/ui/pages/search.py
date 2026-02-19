"""
EcoloGRAPH — Search Page.

Hybrid search (BM25 + semantic + LIKE fallback) across indexed papers.
Results are clickable and navigate to the paper detail view.
"""
import streamlit as st
from src.ui.theme import inject_css, result_card


def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">🔍 Search</div>'
        '<div class="hero-subtitle">Search papers, species, and knowledge across your database</div>',
        unsafe_allow_html=True,
    )

    # --- Search mode tabs ---
    tab_papers, tab_species = st.tabs(["📄 Papers", "🧬 Species"])

    with tab_papers:
        _render_paper_search()

    with tab_species:
        _render_species_search()


def _render_species_search():
    """Search species in Neo4j knowledge graph."""
    query = st.text_input(
        "Species search",
        placeholder="e.g. Quercus, Pinus, cod, springtail, wolf",
        label_visibility="collapsed",
        key="species_search_input",
    )

    if not query:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3rem">'
            '<div style="font-size:3rem;margin-bottom:1rem">🧬</div>'
            '<div style="color:#94a3b8">Search by scientific or common name (Neo4j)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        from src.graph.graph_builder import GraphBuilder
        graph = GraphBuilder()

        # Search in both scientific_name AND common_names
        cypher = """
        MATCH (s:Species)
        WHERE toLower(s.scientific_name) CONTAINS toLower($query)
           OR toLower(COALESCE(s.common_names, '')) CONTAINS toLower($query)
        OPTIONAL MATCH (p:Paper)-[:MENTIONS]->(s)
        RETURN
            s.scientific_name AS species,
            s.common_names AS common_name,
            s.family AS family,
            COUNT(DISTINCT p) AS papers,
            COLLECT(DISTINCT {title: p.title, doc_id: p.doc_id})[0..5] AS paper_list
        ORDER BY papers DESC
        LIMIT 50
        """

        with graph._driver.session(database="neo4j") as session:
            result = session.run(cypher, {"query": query})
            species_list = [dict(r) for r in result]

        graph.close()

        if not species_list:
            st.warning(f'No species matching "{query}" found in Neo4j.')
            return

        st.markdown(f'**{len(species_list)} species** matching *"{query}"*')
        st.markdown("---")

        for i, sp in enumerate(species_list):
            common = sp.get("common_name") or ""
            family = sp.get("family") or ""
            papers_count = sp.get("papers", 0)
            paper_list = sp.get("paper_list", [])

            header_parts = [
                f'<span style="color:#10b981;font-weight:700;font-style:italic">{sp["species"]}</span>'
            ]
            if common:
                header_parts.append(f'<span style="color:#94a3b8"> — {common}</span>')

            badges = ""
            if family:
                badges += f'<span class="domain-badge secondary">{family}</span>'
            badges += f'<span class="domain-badge">{papers_count} paper{"s" if papers_count != 1 else ""}</span>'

            st.markdown(
                f'<div class="glass-card" style="padding:0.8rem 1rem;margin-bottom:0.4rem">'
                f'<div>{"".join(header_parts)}</div>'
                f'<div style="display:flex;gap:0.4rem;margin-top:0.3rem">{badges}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Clickable paper links
            if paper_list:
                for j, paper in enumerate(paper_list):
                    if paper and paper.get("title"):
                        if st.button(
                            f"📄 {paper['title'][:60]}{'...' if len(paper.get('title','')) > 60 else ''}",
                            key=f"sp_{i}_paper_{j}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_paper_id = paper.get("doc_id")
                            st.session_state.nav_page = "📄 Papers"
                            st.rerun()

    except Exception as e:
        st.error(f"Species search failed: {e}")
        st.caption("Make sure Neo4j is running and papers have been ingested.")


def _render_paper_search():
    """Render the paper search interface."""

    # --- Controls ---
    query = st.text_input(
        "Search query",
        placeholder="e.g. microplastics impact on marine fish populations",
        label_visibility="collapsed",
        key="paper_search_input",
    )

    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        limit = st.selectbox("Max results", [10, 20, 50, 100], index=0, label_visibility="collapsed")
    with col_opts2:
        domain_filter = st.selectbox(
            "Filter by domain",
            ["All domains"] + _get_domain_list(),
            index=0,
            label_visibility="collapsed",
        )

    if query:
        with st.spinner("Searching..."):
            results = _search(query, limit, domain_filter)

        if results:
            st.markdown(f"**{len(results)} results** for *\"{query}\"*")
            st.markdown("---")

            for i, r in enumerate(results):
                col_card, col_btn = st.columns([6, 1])
                with col_card:
                    st.markdown(
                        result_card(
                            title=r.get("title", "Untitled"),
                            score=r.get("score"),
                            domain=r.get("domain"),
                            year=r.get("year"),
                            snippet=r.get("snippet"),
                        ),
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("📖", key=f"search_view_{i}", help="View paper"):
                        st.session_state.selected_paper_id = r.get("doc_id")
                        st.session_state.nav_page = "📄 Papers"
                        st.rerun()
        else:
            st.warning(f"No results found for \"{query}\"")
    else:
        # Show placeholder
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:3rem">'
            '<div style="font-size:3rem;margin-bottom:1rem">📚</div>'
            '<div style="color:#94a3b8">Enter a search query to find papers in your knowledge base</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def _search(query, limit, domain_filter):
    """Execute search using RankedSearch, then FTS, then LIKE fallback."""
    results = []

    # --- Strategy 1: RankedSearch (hybrid BM25 + semantic) ---
    try:
        from src.search.ranked_search import RankedSearch
        from src.search.paper_index import PaperIndex

        idx = PaperIndex()
        search = RankedSearch(paper_index=idx)

        if domain_filter and domain_filter != "All domains":
            raw = search.search_by_domain(query=query, domain=domain_filter, limit=limit)
        else:
            raw = search.search(query=query, limit=limit)

        for r in raw:
            results.append({
                "doc_id": r.doc_id,
                "title": r.title,
                "score": r.combined_score,
                "domain": r.primary_domain,
                "year": r.year,
                "snippet": r.snippet,
            })
        if results:
            return results
    except Exception:
        pass

    # --- Strategy 2: PaperIndex FTS5 ---
    try:
        from src.search.paper_index import PaperIndex
        idx = PaperIndex()
        raw = idx.search(query, limit=limit)
        for r in raw:
            results.append({
                "doc_id": r.doc_id,
                "title": r.title,
                "score": abs(r.score) if r.score else None,
                "domain": r.primary_domain if hasattr(r, 'primary_domain') else None,
                "year": r.year if hasattr(r, 'year') else None,
                "snippet": r.snippet,
            })
        if results:
            return results
    except Exception:
        pass

    # --- Strategy 3: LIKE fallback (catches partial matches FTS misses) ---
    try:
        from src.search.paper_index import PaperIndex
        idx = PaperIndex()
        raw = _search_like_fallback(idx, query, limit, domain_filter)
        return raw
    except Exception as e:
        st.error(f"Search infrastructure not available: {e}")
        return []


def _search_like_fallback(idx, query, limit, domain_filter):
    """SQL LIKE search as fallback when FTS5 doesn't match well."""
    import sqlite3

    with sqlite3.connect(idx.db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Build LIKE conditions for each query token
        tokens = query.strip().split()
        if not tokens:
            return []

        conditions = []
        params = []
        for token in tokens:
            pattern = f"%{token}%"
            conditions.append(
                "(p.title LIKE ? OR p.abstract LIKE ? OR p.keywords LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])

        where = " OR ".join(conditions)
        sql = f"""
            SELECT p.doc_id, p.title, p.year, p.primary_domain, p.abstract
            FROM papers p
            WHERE ({where})
        """

        if domain_filter and domain_filter != "All domains":
            sql += " AND p.primary_domain = ?"
            params.append(domain_filter)

        sql += " LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        results = []
        for row in cursor:
            abstract = row["abstract"] or ""
            snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
            results.append({
                "doc_id": row["doc_id"],
                "title": row["title"],
                "score": None,
                "domain": row["primary_domain"],
                "year": row["year"],
                "snippet": snippet,
            })
        return results


def _get_domain_list():
    """Get list of available domain names."""
    try:
        from src.core.domain_registry import DomainType
        return sorted([d.value for d in DomainType if d.value != "unknown"])
    except Exception:
        return []
