"""
Species Taxonomy Explorer — Validate, resolve, and explore species taxonomy.

Features:
1. Browse all extracted species from Neo4j
2. Look up any species (scientific or common name) via GBIF
3. Resolve common names → scientific names
4. View taxonomic hierarchy for validated species
"""
import streamlit as st
import pandas as pd

from src.graph.graph_builder import GraphBuilder
from src.scrapers.gbif_occurrence_client import GBIFOccurrenceClient
from src.core.config import get_settings
from src.ui.theme import inject_css


# ── Cached clients ──────────────────────────────────

@st.cache_resource
def get_graph_builder():
    try:
        settings = get_settings()
        return GraphBuilder(
            uri=settings.neo4j.uri,
            username=settings.neo4j.user,
            password=settings.neo4j.password
        )
    except Exception as e:
        return None


@st.cache_resource
def get_gbif_client():
    return GBIFOccurrenceClient()


# ── Neo4j queries ───────────────────────────────────

def fetch_species_from_graph():
    """Fetch all species from Neo4j graph."""
    graph = get_graph_builder()
    if graph is None:
        return pd.DataFrame()

    query = """
    MATCH (s:Species)
    OPTIONAL MATCH (p:Paper)-[:MENTIONS]->(s)
    RETURN 
        s.scientific_name AS species_name,
        s.common_names AS common_name,
        s.family AS family,
        s.gbif_validated AS validated,
        COUNT(DISTINCT p) AS paper_count,
        COLLECT(DISTINCT p.title)[0..3] AS sample_papers
    ORDER BY paper_count DESC
    """
    with graph._driver.session(database="neo4j") as session:
        result = session.run(query)
        data = []
        for r in result:
            data.append({
                "species_name": r["species_name"],
                "common_name": r["common_name"] or "-",
                "family": r["family"] or "-",
                "validated": "✅" if r["validated"] else "—",
                "paper_count": r["paper_count"],
                "sample_papers": r["sample_papers"]
            })
    return pd.DataFrame(data)


# ── Render ──────────────────────────────────────────

def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">🧬 Taxonomy Explorer</div>'
        '<div class="hero-subtitle">Validate species names, resolve common names, and explore taxonomy via GBIF</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📋 Database Species", "🔍 Name Resolver", "📊 Taxonomy Stats"])

    # ── Tab 1: Browse extracted species ─────────────
    with tab1:
        st.subheader("Species from Your Papers")

        with st.spinner("Loading species from Neo4j..."):
            try:
                df = fetch_species_from_graph()
                if df.empty:
                    st.info("No species found. Ingest papers first.")
                else:
                    st.success(f"**{len(df)}** unique species extracted")

                    # Filters
                    col1, col2 = st.columns(2)
                    with col1:
                        min_p = st.number_input("Min papers", 1, int(df["paper_count"].max()), 1)
                    with col2:
                        search = st.text_input("Filter by name", "", key="val_search")

                    filtered = df[df["paper_count"] >= min_p]
                    if search:
                        mask = filtered["species_name"].str.contains(search, case=False, na=False)
                        filtered = filtered[mask]

                    st.dataframe(filtered, use_container_width=True, height=400)

                    csv = filtered.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "species_list.csv", "text/csv")

            except Exception as e:
                st.error(f"Error loading species: {e}")
                st.info("Make sure Neo4j is running and papers have been ingested.")

    # ── Tab 2: Name Resolver ────────────────────────
    with tab2:
        st.subheader("🔍 Resolve Any Name → Scientific Name")
        st.caption(
            "Enter a common name (e.g. 'cod', 'grey wolf', 'springtail') or partial scientific name. "
            "GBIF will attempt to resolve it to a canonical scientific name."
        )

        query = st.text_input("Species or common name", placeholder="e.g. Atlantic cod, springtail, Quercus", key="resolver_input")

        if query:
            gbif = get_gbif_client()

            with st.spinner(f"Resolving '{query}' via GBIF..."):
                result = gbif.validate_species(query)

            if result:
                match_type = result.get("matchType", "NONE")
                confidence = result.get("confidence", 0)

                # Header card
                if match_type == "EXACT":
                    badge_color = "#10b981"
                    badge_text = "EXACT MATCH"
                elif match_type == "FUZZY":
                    badge_color = "#f59e0b"
                    badge_text = "FUZZY MATCH"
                else:
                    badge_color = "#ef4444"
                    badge_text = match_type

                st.markdown(
                    f'<div class="glass-card" style="padding:1.2rem">'
                    f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem">'
                    f'<span style="font-size:1.4rem;font-weight:700;color:#10b981;font-style:italic">{result["canonical_name"]}</span>'
                    f'<span style="background:{badge_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem">{badge_text}</span>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem;color:#94a3b8;font-size:0.85rem">'
                    f'<div><b>Rank:</b> {result["rank"]}</div>'
                    f'<div><b>Family:</b> {result["family"]}</div>'
                    f'<div><b>Kingdom:</b> {result["kingdom"]}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if result.get("taxon_key"):
                    st.markdown(f"🔗 [View on GBIF](https://www.gbif.org/species/{result['taxon_key']})")

                # Check if in Neo4j
                df = fetch_species_from_graph()
                if not df.empty:
                    matches = df[df["species_name"].str.lower() == result["canonical_name"].lower()]
                    if not matches.empty:
                        st.success(f"✅ This species is in your database ({matches.iloc[0]['paper_count']} papers)")
                    else:
                        st.info("ℹ️ This species is not yet in your Neo4j database")

                # Batch resolve from database
                st.markdown("---")
                if st.button("🔄 Batch-resolve all database species", help="Validate all Neo4j species against GBIF"):
                    _batch_resolve()
            else:
                st.warning(f"No match found for '{query}'. Try a different name or spelling.")

    # ── Tab 3: Taxonomy Stats ───────────────────────
    with tab3:
        st.subheader("Taxonomic Breakdown")

        df = fetch_species_from_graph()
        if df.empty:
            st.info("No species data. Ingest papers first.")
        else:
            # Family breakdown
            families = df[df["family"] != "-"]["family"].value_counts().head(15)
            if not families.empty:
                st.markdown("### Top Families")
                st.bar_chart(families)

            # Validation status
            validated_count = len(df[df["validated"] == "✅"])
            not_validated = len(df) - validated_count

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Species", len(df))
            with col2:
                st.metric("GBIF Validated", validated_count)
            with col3:
                st.metric("Not Validated", not_validated)

            # Papers per species distribution
            st.markdown("### Papers per Species")
            hist_data = df["paper_count"].value_counts().sort_index().head(20)
            st.bar_chart(hist_data)


def _batch_resolve():
    """Batch-resolve all database species against GBIF."""
    df = fetch_species_from_graph()
    if df.empty:
        st.warning("No species to resolve.")
        return

    gbif = get_gbif_client()
    graph = get_graph_builder()
    if graph is None:
        st.error("Neo4j not available.")
        return

    progress = st.progress(0)
    status = st.empty()
    results = []

    for idx, row in df.iterrows():
        name = row["species_name"]
        status.text(f"Resolving {idx + 1}/{len(df)}: {name}")

        validation = gbif.validate_species(name)
        if validation:
            results.append({
                "original": name,
                "canonical": validation["canonical_name"],
                "match": validation["matchType"],
                "family": validation["family"],
                "kingdom": validation["kingdom"],
                "confidence": validation["confidence"],
            })

            # Update Neo4j
            try:
                update_q = """
                MATCH (s:Species {scientific_name: $original})
                SET s.gbif_validated = true,
                    s.gbif_taxon_key = $taxon_key,
                    s.family = $family,
                    s.rank = $rank
                """
                with graph._driver.session(database="neo4j") as session:
                    session.run(update_q,
                                original=name,
                                taxon_key=validation.get("taxon_key"),
                                family=validation["family"],
                                rank=validation["rank"])
            except Exception:
                pass

        progress.progress((idx + 1) / len(df))

    status.text("✅ Batch resolution complete!")
    progress.empty()

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        st.cache_resource.clear()
