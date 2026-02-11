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
        '<div class="hero-title">🔍 Paper Search</div>'
        '<div class="hero-subtitle">Hybrid BM25 + Semantic search across your knowledge base</div>',
        unsafe_allow_html=True,
    )

    # --- Controls ---
    query = st.text_input(
        "Search query",
        placeholder="e.g. microplastics impact on marine fish populations",
        label_visibility="collapsed",
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
