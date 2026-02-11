"""
Export utilities for EcoloGRAPH.

Provides functions to export data in various formats:
- Graph visualizations to PNG
- Chunks to JSON
- Papers to BibTeX
"""
import json
import base64
from io import BytesIO
from typing import List, Dict
import streamlit as st


def export_graph_as_png(fig, filename="graph_export.png"):
    """
    Export a matplotlib/plotly figure as PNG.
    
    Args:
        fig: Matplotlib or Plotly figure object
        filename: Name for downloaded file
    """
    try:
        # Try plotly first
        if hasattr(fig, 'to_image'):
            img_bytes = fig.to_image(format="png")
        # Then matplotlib
        elif hasattr(fig, 'savefig'):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            img_bytes = buf.getvalue()
            buf.close()
        else:
            st.error("Unsupported figure type for export")
            return
            
        # Create download button
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download Graph PNG</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Failed to export graph: {e}")


def export_chunks_as_json(chunks: List[Dict], filename="chunks_export.json"):
    """
    Export chunks as JSON file.
    
    Args:
        chunks: List of chunk dictionaries
        filename: Name for downloaded file
    """
    try:
        json_str = json.dumps(chunks, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📦 Download Chunks (JSON)",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Failed to export chunks: {e}")


def export_papers_as_bibtex(papers: List[Dict], filename="papers_export.bib"):
    """
    Export papers as BibTeX file.
    
    Args:
        papers: List of paper dictionaries with metadata
        filename: Name for downloaded file
    """
    try:
        bibtex_entries = []
        
        for i, paper in enumerate(papers):
            # Generate citation key
            first_author = paper.get('authors', ['Unknown'])[0].split()[-1] if paper.get('authors') else 'Unknown'
            year = paper.get('year', 'XXXX')
            cite_key = f"{first_author}{year}_{i}"
            
            # Build BibTeX entry
            entry = f"@article{{{cite_key},\n"
            
            if paper.get('title'):
                entry += f"  title = {{{paper['title']}}},\n"
            
            if paper.get('authors'):
                authors_str = " and ".join(paper['authors'][:5])  # Limit to 5 authors
                entry += f"  author = {{{authors_str}}},\n"
            
            if paper.get('year'):
                entry += f"  year = {{{paper['year']}}},\n"
            
            if paper.get('journal'):
                entry += f"  journal = {{{paper['journal']}}},\n"
            
            if paper.get('doi'):
                entry += f"  doi = {{{paper['doi']}}},\n"
            
            if paper.get('abstract'):
                # Clean abstract for BibTeX
                abstract_clean = paper['abstract'].replace('\n', ' ').replace('{', '').replace('}', '')
                entry += f"  abstract = {{{abstract_clean}}},\n"
            
            entry += "}\n"
            bibtex_entries.append(entry)
        
        bibtex_str = "\n".join(bibtex_entries)
        
        st.download_button(
            label="📚 Download Papers (BibTeX)",
            data=bibtex_str,
            file_name=filename,
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"Failed to export BibTeX: {e}")


def export_button_row(
    chunks: List[Dict] = None,
    papers: List[Dict] = None,
    graph_fig = None
):
    """
    Render a row of export buttons for available data.
    
    Args:
        chunks: Optional list of chunks to export
        papers: Optional list of papers to export
        graph_fig: Optional graph figure to export
    """
    cols = []
    active_exports = sum([chunks is not None, papers is not None, graph_fig is not None])
    
    if active_exports == 0:
        return
    
    cols = st.columns(active_exports)
    col_idx = 0
    
    with st.expander("📤 Export Options", expanded=False):
        export_cols = st.columns(active_exports)
        
        if graph_fig is not None:
            with export_cols[col_idx]:
                st.markdown("**Graph Visualization**")
                export_graph_as_png(graph_fig)
            col_idx += 1
        
        if chunks is not None and len(chunks) > 0:
            with export_cols[col_idx]:
                st.markdown("**Document Chunks**")
                export_chunks_as_json(chunks)
            col_idx += 1
        
        if papers is not None and len(papers) > 0:
            with export_cols[col_idx]:
                st.markdown("**Paper Metadata**")
                export_papers_as_bibtex(papers)
            col_idx += 1
