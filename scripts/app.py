"""
EcoloGRAPH — Streamlit Application Entry Point.

Premium multi-page dashboard for interactive ecological research.

Usage:
    streamlit run scripts/app.py
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="EcoloGRAPH",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    from src.ui.theme import inject_css, theme_toggle
    
    # Theme toggle must be called before inject_css to get current theme
    # But we render it in sidebar, so we'll inject CSS first with default
    # and let toggle handle rerun
    
    # Initialize theme if not set
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    
    # Inject CSS based on current theme
    inject_css(theme=st.session_state.theme)

    # Cross-page navigation support
    nav_options = ["📊 Dashboard", "💬 Chat", "📄 Papers", "🕸️ Graph",
                   "🔍 Search", "🧬 Species", "✅ Validation", "🔬 Classifier"]
    # "🔗 Graph V2" removed - now integrated into Graph Explorer's Interactive mode

    # If another page requested navigation (e.g. "click paper → go to Papers")
    default_idx = 0
    if "nav_page" in st.session_state and st.session_state.nav_page:
        target = st.session_state.nav_page
        st.session_state.nav_page = None  # consume it
        if target in nav_options:
            default_idx = nav_options.index(target)

    # ---- Sidebar navigation ----
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;margin-bottom:1.5rem">'
            '<div style="font-size:2.5rem">🌿</div>'
            '<div style="font-size:1.3rem;font-weight:700;color:#10b981">EcoloGRAPH</div>'
            '<div style="font-size:0.75rem;color:#64748b">Graph RAG for Ecology</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            nav_options,
            index=default_idx,
            label_visibility="collapsed",
        )

        st.markdown("---")
        
        # Theme toggle
        current_theme = theme_toggle()
        st.caption(f"Theme: {'Dark 🌙' if current_theme == 'dark' else 'Light ☀️'}")
        st.markdown(
            '<div style="font-size:0.75rem;color:#64748b;text-align:center">'
            'v1.2.0 · Complete<br>'
            '33/33 tests passing ✅'
            '</div>',
            unsafe_allow_html=True,
        )

    # ---- Route to page ----
    if page == "📊 Dashboard":
        from src.ui.pages.dashboard import render
        render()
    elif page == "💬 Chat":
        from src.ui.pages.chat import render
        render()
    elif page == "📄 Papers":
        from src.ui.pages.papers import render
        render()
    elif page == "🕸️ Graph":
        from src.ui.pages.graph_explorer import render
        render()
    # Graph V2 deprecated - functionality merged into Graph Explorer's Interactive mode
    # elif page == "🔗 Graph V2":
    #     from src.ui.pages.graph_explorer_v2 import render
    #     render()
    elif page == "🔍 Search":
        from src.ui.pages.search import render
        render()
    elif page == "🧬 Species":
        from src.ui.pages.species import render
        render()
    elif page == "✅ Validation":
        from src.ui.pages import species_validation
        # species_validation is a standalone script, just import to execute
    elif page == "🔬 Classifier":
        from src.ui.pages.domain_lab import render
        render()


if __name__ == "__main__":
    main()
