"""
EcoloGRAPH — Species Explorer Page.

Look up species information using the free GBIF API.
Shows taxonomy, distribution, occurrence records, and conservation status.
No API token required.
"""
import streamlit as st
from src.ui.theme import inject_css


def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">🔬 Species Explorer</div>'
        '<div class="hero-subtitle">Look up taxonomy, distribution, and occurrence data from GBIF (free, no API key)</div>',
        unsafe_allow_html=True,
    )

    # --- Input ---
    col1, col2 = st.columns([3, 1])
    with col1:
        species = st.text_input(
            "Species name",
            placeholder="e.g. Gadus morhua, Canis lupus, Quercus robur",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("🔍 Look up", use_container_width=True)

    if species and search_btn:
        _show_species_info(species)
    elif not species:
        # Featured species showcase
        st.markdown("---")
        st.subheader("🌟 Try these species")
        cols = st.columns(4)
        examples = [
            ("🐟", "Gadus morhua", "Atlantic cod"),
            ("🐺", "Canis lupus", "Grey wolf"),
            ("🌳", "Quercus robur", "European oak"),
            ("🐢", "Chelonia mydas", "Green sea turtle"),
        ]
        for col, (icon, sci, common) in zip(cols, examples):
            with col:
                st.markdown(
                    f'<div class="glass-card" style="text-align:center">'
                    f'<div style="font-size:1.8rem">{icon}</div>'
                    f'<div style="color:#e2e8f0;font-weight:600;margin:0.3rem 0;font-style:italic">{sci}</div>'
                    f'<div style="color:#64748b;font-size:0.85rem">{common}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _show_species_info(species_name):
    """Display species information from GBIF."""
    st.markdown("---")

    tab_overview, tab_dist, tab_records = st.tabs([
        "📋 Overview & Taxonomy",
        "🌍 Distribution Map",
        "📊 Occurrence Records",
    ])

    with tab_overview:
        _render_overview(species_name)

    with tab_dist:
        _render_distribution(species_name)

    with tab_records:
        _render_records(species_name)


def _render_overview(species_name):
    """Render species overview: taxonomy + basic info from GBIF Species API."""
    import httpx

    with st.spinner("Querying GBIF Species API..."):
        try:
            with httpx.Client(timeout=15.0) as client:
                # Match species name
                resp = client.get(
                    "https://api.gbif.org/v1/species/match",
                    params={"name": species_name, "verbose": "true"},
                )
                if resp.status_code != 200:
                    st.error("GBIF Species API returned an error.")
                    return

                match = resp.json()

                if match.get("matchType") == "NONE":
                    st.warning(f"No species found for **{species_name}**. Check the spelling.")
                    return

                species_key = match.get("usageKey")

                # Get full species detail
                detail = {}
                if species_key:
                    resp2 = client.get(f"https://api.gbif.org/v1/species/{species_key}")
                    if resp2.status_code == 200:
                        detail = resp2.json()

                    # Get vernacular names
                    resp3 = client.get(
                        f"https://api.gbif.org/v1/species/{species_key}/vernacularNames",
                        params={"limit": 20},
                    )
                    vernacular = []
                    if resp3.status_code == 200:
                        for v in resp3.json().get("results", []):
                            name = v.get("vernacularName")
                            lang = v.get("language", "")
                            if name:
                                vernacular.append({"name": name, "language": lang})

        except Exception as e:
            st.error(f"Error querying GBIF: {e}")
            return

    # --- Species header card ---
    canonical = match.get("canonicalName", species_name)
    author = detail.get("authorship", match.get("authorship", ""))
    status = match.get("status", "")
    rank = match.get("rank", "SPECIES")

    st.markdown(
        f'<div class="glass-card">'
        f'<div style="font-size:1.4rem;font-weight:700;color:#10b981;font-style:italic">{canonical}</div>'
        f'<div style="color:#94a3b8;margin-top:0.2rem">{author}</div>'
        f'<div style="display:flex;gap:0.5rem;margin-top:0.5rem">'
        f'<span class="domain-badge">{rank.title()}</span>'
        f'<span class="domain-badge secondary">{status.replace("_", " ").title()}</span>'
        f'<span class="domain-badge secondary">Match: {match.get("matchType", "UNKNOWN")}</span>'
        f'<span class="domain-badge secondary">Confidence: {match.get("confidence", 0)}%</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # --- Taxonomy table ---
    st.markdown("### 🧬 Taxonomy")

    taxonomy_ranks = [
        ("Kingdom", match.get("kingdom")),
        ("Phylum", match.get("phylum")),
        ("Class", match.get("class")),
        ("Order", match.get("order")),
        ("Family", match.get("family")),
        ("Genus", match.get("genus")),
        ("Species", match.get("species")),
    ]

    import pandas as pd
    tax_df = pd.DataFrame(
        [(rank, name) for rank, name in taxonomy_ranks if name],
        columns=["Rank", "Name"],
    )
    st.dataframe(tax_df, use_container_width=True, hide_index=True)

    # --- Vernacular names ---
    if vernacular:
        st.markdown("### 🗣️ Common Names")
        name_df = pd.DataFrame(vernacular[:15])
        name_df.columns = ["Common Name", "Language"]
        st.dataframe(name_df, use_container_width=True, hide_index=True)

    # --- IUCN status from GBIF (if available) ---
    iucn_cat = detail.get("iucnRedListCategory") or match.get("iucnRedListCategory")
    if iucn_cat:
        st.markdown("### 🛡️ Conservation Status")
        cat_names = {
            "EX": "Extinct", "EW": "Extinct in the Wild",
            "CR": "Critically Endangered", "EN": "Endangered",
            "VU": "Vulnerable", "NT": "Near Threatened",
            "LC": "Least Concern", "DD": "Data Deficient", "NE": "Not Evaluated",
        }
        cat_colors = {
            "CR": "#ef4444", "EN": "#f97316", "VU": "#f59e0b",
            "NT": "#3b82f6", "LC": "#10b981", "DD": "#94a3b8",
        }
        color = cat_colors.get(iucn_cat, "#94a3b8")
        name = cat_names.get(iucn_cat, iucn_cat)

        st.markdown(
            f'<div class="glass-card" style="text-align:center">'
            f'<div style="font-size:2.5rem;font-weight:700;color:{color}">{iucn_cat}</div>'
            f'<div style="color:#94a3b8;font-size:1.1rem">{name}</div>'
            f'<div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem">Source: GBIF Species API</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Extra detail ---
    if detail:
        extra = {}
        for key in ["habitat", "nomenclaturalStatus", "taxonomicStatus", "numDescendants", "nubKey"]:
            val = detail.get(key)
            if val is not None:
                extra[key.replace("n", " n").replace("N", " N").strip().title()] = val
        if extra:
            st.markdown("### 📊 Additional Details")
            st.json(extra)


def _render_distribution(species_name):
    """Render distribution map from GBIF occurrences."""
    try:
        from src.scrapers.gbif_occurrence_client import GBIFOccurrenceClient
        with st.spinner("Fetching occurrence data from GBIF..."):
            with GBIFOccurrenceClient() as gbif:
                dist = gbif.get_distribution(species_name, sample_size=50)

        if not dist or dist.total_occurrences == 0:
            st.warning(f"No GBIF occurrence records for *{species_name}*.")
            return

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Occurrences", f"{dist.total_occurrences:,}")
        c2.metric("Georeferenced", f"{dist.occurrences_with_coords:,}")
        if dist.min_year and dist.max_year:
            c3.metric("Year Range", f"{dist.min_year}–{dist.max_year}")

        # Map
        if dist.sample_records:
            import pandas as pd
            map_data = pd.DataFrame([
                {"lat": r.decimal_latitude, "lon": r.decimal_longitude}
                for r in dist.sample_records
                if r.decimal_latitude is not None and r.decimal_longitude is not None
            ])
            if not map_data.empty:
                st.markdown("### 🗺️ Occurrence Map")
                st.map(map_data, size=20, color="#10b981")

        # Countries
        if dist.countries:
            st.markdown("### 🌐 Countries with Records")
            import pandas as pd
            country_df = pd.DataFrame(
                [(c, "") for c in dist.countries[:30]],
                columns=["Country", ""],
            )
            badges = "".join(
                f'<span class="domain-badge secondary">{c}</span>' for c in dist.countries[:30]
            )
            st.markdown(badges, unsafe_allow_html=True)

            if len(dist.countries) > 30:
                st.caption(f"... and {len(dist.countries) - 30} more countries")

    except Exception as e:
        st.error(f"GBIF error: {e}")


def _render_records(species_name):
    """Render detailed occurrence records table."""
    try:
        from src.scrapers.gbif_occurrence_client import GBIFOccurrenceClient
        with st.spinner("Fetching occurrence records..."):
            with GBIFOccurrenceClient() as gbif:
                records = gbif.get_occurrences(
                    scientific_name=species_name,
                    has_coordinate=True,
                    limit=100,
                )

        if not records:
            st.warning(f"No occurrence records found for *{species_name}*.")
            return

        st.markdown(f"### 📊 Occurrence Records ({len(records)} shown)")

        import pandas as pd
        df = pd.DataFrame([
            {
                "Scientific Name": r.scientific_name,
                "Latitude": r.decimal_latitude,
                "Longitude": r.decimal_longitude,
                "Country": r.country,
                "Date": r.event_date,
                "Year": r.year,
                "Depth (m)": r.depth_meters,
                "Elevation (m)": r.elevation_meters,
                "Institution": r.institution_code,
                "Basis": (r.basis_of_record or "").replace("_", " ").title(),
            }
            for r in records
        ])

        # Remove columns that are entirely empty
        df = df.dropna(axis=1, how="all")

        st.dataframe(df, use_container_width=True, hide_index=True, height=500)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download as CSV",
            csv,
            file_name=f"gbif_{species_name.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"GBIF error: {e}")
