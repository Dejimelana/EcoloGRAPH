"""
Species Validation UI - Interactive GBIF validation and refinement tool.

Allows users to:
1. View all extracted species from database
2. Validate species names against GBIF Backbone Taxonomy
3. Filter out invalid/hallucinated species
4. Normalize species names to canonical forms
5. Export cleaned species list
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.graph_builder import GraphBuilder
from src.scrapers.gbif_occurrence_client import GBIFOccurrenceClient
from src.core.config import get_settings
from src.ui.theme import inject_css

# Apply theme
inject_css()

st.set_page_config(
    page_title="Species Validation - EcoloGRAPH",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Species Validation")
st.markdown("Validate extracted species against GBIF Backbone Taxonomy")

# Initialize clients
@st.cache_resource
def get_graph_builder():
    """Initialize GraphBuilder with error handling for when Neo4j is not available."""
    try:
        settings = get_settings()
        return GraphBuilder(
            uri=settings.neo4j.uri,
            username=settings.neo4j.user,
            password=settings.neo4j.password
        )
    except Exception as e:
        st.error(f"Cannot connect to Neo4j: {e}")
        st.info("Make sure Neo4j is running with: `docker start neo4j`")
        return None

@st.cache_resource
def get_gbif_client():
    return GBIFOccurrenceClient()


def fetch_species_from_graph():
    """Fetch all species from Neo4j graph."""
    graph = get_graph_builder()
    
    if graph is None:
        return pd.DataFrame()  # Return empty dataframe if Neo4j unavailable
    
    query = """
    MATCH (s:Species)
    OPTIONAL MATCH (s)-[:MENTIONED_IN]->(p:Paper)
    RETURN 
        s.scientific_name AS species_name,
        s.common_name AS common_name,
        COUNT(DISTINCT p) AS paper_count,
        COLLECT(DISTINCT p.title)[0..3] AS sample_papers
    ORDER BY paper_count DESC
    """
    
    with graph._driver.session(database="neo4j") as session:
        result = session.run(query)
        
        species_data = []
        for record in result:
            species_data.append({
                "species_name": record["species_name"],
                "common_name": record["common_name"] or "-",
                "paper_count": record["paper_count"],
                "sample_papers": record["sample_papers"]
            })
    
    return pd.DataFrame(species_data)


def validate_species_with_gbif(species_name: str) -> dict:
    """Validate species against GBIF."""
    gbif = get_gbif_client()
    
    result = gbif.validate_species(species_name)
    
    if result:
        return {
            "status": "✅ Valid",
            "canonical_name": result.get("canonical_name", species_name),
            "rank": result.get("rank", "Unknown"),
            "family": result.get("family", "-"),
            "kingdom": result.get("kingdom", "-"),
            "match_type": result.get("matchType", "EXACT"),
            "taxon_key": result.get("taxon_key"),
            "confidence": "HIGH" if result.get("matchType") == "EXACT" else "MEDIUM"
        }
    else:
        return {
            "status": "❌ Invalid",
            "canonical_name": species_name,
            "rank": "Unknown",
            "family": "-",
            "kingdom": "-",
            "match_type": "NOT_FOUND",
            "taxon_key": None,
            "confidence": "NONE"
        }


# Main UI
tab1, tab2, tab3 = st.tabs(["📋 Species List", "🔍 Validate", "📊 Summary"])

with tab1:
    st.subheader("Extracted Species from Database")
    
    with st.spinner("Loading species from Neo4j..."):
        try:
            df_species = fetch_species_from_graph()
            
            if df_species.empty:
                st.info("No species found in the graph database. Ingest some papers first.")
            else:
                st.success(f"Found **{len(df_species)}** unique species")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    min_papers = st.number_input(
                        "Min papers",
                        min_value=1,
                        max_value=int(df_species['paper_count'].max()),
                        value=1
                    )
                with col2:
                    search_term = st.text_input("Search species", "")
                
                # Apply filters
                filtered_df = df_species[df_species['paper_count'] >= min_papers]
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df['species_name'].str.contains(search_term, case=False, na=False)
                    ]
                
                st.dataframe(
                    filtered_df,
                    width='stretch',
                    height=400
                )
                
                # Export button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Species List (CSV)",
                    data=csv,
                    file_name="species_list.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error loading species: {e}")
            st.info("Make sure Neo4j is running and papers have been ingested.")

with tab2:
    st.subheader("GBIF Validation")
    
    st.markdown("""
    Click **Validate All** to check each species against GBIF Backbone Taxonomy.
    This process may take a few minutes for large datasets.
    """)
    
    if st.button("🔄 Validate All Species", type="primary"):
        try:
            df_species = fetch_species_from_graph()
            
            if df_species.empty:
                st.warning("No species to validate.")
            else:
                validation_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df_species.iterrows():
                    species_name = row['species_name']
                    status_text.text(f"Validating {idx+1}/{len(df_species)}: {species_name}")
                    
                    validation = validate_species_with_gbif(species_name)
                    validation_results.append({
                        "original_name": species_name,
                        **validation,
                        "paper_count": row['paper_count']
                    })
                    
                    progress_bar.progress((idx + 1) / len(df_species))
                
                df_validation = pd.DataFrame(validation_results)
                st.session_state['validation_results'] = df_validation
                
                status_text.text("✅ Validation complete!")
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"Validation error: {e}")
    
    # Show validation results
    if 'validation_results' in st.session_state:
        df_val = st.session_state['validation_results']
        
        st.markdown("---")
        st.subheader("Validation Results")
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            valid_count = len(df_val[df_val['status'] == '✅ Valid'])
            st.metric("Valid Species", valid_count)
        with col2:
            invalid_count = len(df_val[df_val['status'] == '❌ Invalid'])
            st.metric("Invalid Species", invalid_count)
        with col3:
            normalized_count = len(df_val[df_val['original_name'] != df_val['canonical_name']])
            st.metric("Normalized", normalized_count)
        with col4:
            validation_rate = (valid_count / len(df_val)) * 100 if len(df_val) > 0 else 0
            st.metric("Validation Rate", f"{validation_rate:.1f}%")
        
        # Filter control
        filter_option = st.radio(
            "Show",
            ["All", "Valid Only", "Invalid Only", "Normalized"],
            horizontal=True
        )
        
        if filter_option == "Valid Only":
            display_df = df_val[df_val['status'] == '✅ Valid']
        elif filter_option == "Invalid Only":
            display_df = df_val[df_val['status'] == '❌ Invalid']
        elif filter_option == "Normalized":
            display_df = df_val[df_val['original_name'] != df_val['canonical_name']]
        else:
            display_df = df_val
        
        st.dataframe(
            display_df[['status', 'original_name', 'canonical_name', 'rank', 'family', 'confidence', 'paper_count']],
            width='stretch',
            height=400
        )
        
        #Export cleaned list
        st.markdown("---")
        st.subheader("Export Cleaned Data")
        
        valid_df = df_val[df_val['status'] == '✅ Valid']
        csv_clean = valid_df[['canonical_name', 'rank', 'family', 'kingdom', 'taxon_key', 'paper_count']].to_csv(index=False)
        
        st.download_button(
            label="📥 Download Valid Species Only (CSV)",
            data=csv_clean,
            file_name="species_validated.csv",
            mime="text/csv"
        )
        
        # Option to update Neo4j
        st.markdown("---")
        st.warning("⚠️ **Update Database** (Experimental)")
        st.markdown("""
        Replace species names in Neo4j with GBIF-validated canonical names.
        **This will modify your graph database.**
        """)
        
        if st.button("🔄 Update Neo4j with Validated Names", type="secondary"):
            graph = get_graph_builder()
            
            update_count = 0
            for _, row in valid_df.iterrows():
                if row['original_name'] != row['canonical_name']:
                    try:
                        query = """
                        MATCH (s:Species {scientific_name: $original})
                        SET s.scientific_name = $canonical,
                            s.gbif_validated = true,
                            s.gbif_taxon_key = $taxon_key,
                            s.family = $family,
                            s.rank = $rank
                        RETURN s.scientific_name as updated_name
                        """
                        with graph._driver.session(database="neo4j") as session:
                            session.run(
                                query,
                                original=row['original_name'],
                                canonical=row['canonical_name'],
                                taxon_key=row['taxon_key'],
                                family=row['family'],
                                rank=row['rank']
                            )
                        update_count += 1
                    except Exception as e:
                        st.error(f"Error updating {row['original_name']}: {e}")
            
            st.success(f"✅ Updated {update_count} species names in Neo4j")
            st.cache_resource.clear()  # Clear cache to reload data

with tab3:
    st.subheader("Validation Summary")
    
    if 'validation_results' in st.session_state:
        df_val = st.session_state['validation_results']
        
        # Summary metrics
        st.metric("Total Species", len(df_val))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Validation Status")
            status_counts = df_val['status'].value_counts()
            st.bar_chart(status_counts)
        
        with col2:
            st.markdown("###Confidence Distribution")
            conf_counts = df_val['confidence'].value_counts()
            st.bar_chart(conf_counts)
        
        # Taxonomic breakdown
        st.markdown("### Taxonomic Breakdown (Valid Species)")
        valid_df = df_val[df_val['status'] == '✅ Valid']
        
        if not valid_df.empty:
            family_counts = valid_df['family'].value_counts().head(10)
            st.bar_chart(family_counts)
        else:
            st.info("No valid species to display taxonomy for.")
            
    else:
        st.info("Run validation first to see summary statistics.")
