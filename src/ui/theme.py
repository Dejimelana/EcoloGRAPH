"""
EcoloGRAPH UI Theme — Premium dark glassmorphism design system.

Provides CSS injection and styling utilities for the Streamlit app.
"""

# Color palette
COLORS = {
    "bg_primary": "#0f1419",
    "bg_secondary": "#1a2332",
    "bg_card": "rgba(26, 35, 50, 0.7)",
    "bg_card_hover": "rgba(26, 35, 50, 0.9)",
    "accent": "#10b981",
    "accent_light": "#34d399",
    "accent_dark": "#059669",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border": "rgba(148, 163, 184, 0.1)",
    "border_accent": "rgba(16, 185, 129, 0.3)",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
    "success": "#10b981",
}

MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Global ---- */
.stApp {
    background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1923 100%);
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.08);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #10b981;
}

/* ---- Cards ---- */
.glass-card {
    background: rgba(26, 35, 50, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(16, 185, 129, 0.3);
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1);
    transform: translateY(-2px);
}

/* ---- Metric Cards ---- */
.metric-card {
    background: rgba(26, 35, 50, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(16, 185, 129, 0.3);
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.08);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #10b981;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

/* ---- Result Cards ---- */
.result-card {
    background: rgba(26, 35, 50, 0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-left: 3px solid #10b981;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.25s ease;
}
.result-card:hover {
    background: rgba(26, 35, 50, 0.8);
    border-left-color: #34d399;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.08);
}
.result-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.result-meta {
    font-size: 0.8rem;
    color: #64748b;
}
.result-snippet {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* ---- Domain Badge ---- */
.domain-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}
.domain-badge.secondary {
    background: rgba(59, 130, 246, 0.12);
    color: #60a5fa;
    border-color: rgba(59, 130, 246, 0.2);
}

/* ---- Score Bar ---- */
.score-bar-container {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}
.score-bar-label {
    width: 180px;
    font-size: 0.85rem;
    color: #94a3b8;
    flex-shrink: 0;
}
.score-bar-track {
    flex: 1;
    height: 8px;
    background: rgba(148, 163, 184, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin: 0 0.8rem;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.score-bar-value {
    width: 50px;
    text-align: right;
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
}

/* ---- Status indicator ---- */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.online { background: #10b981; box-shadow: 0 0 6px #10b981; }
.status-dot.offline { background: #ef4444; }
.status-dot.warning { background: #f59e0b; }

/* ---- Hypothesis Card ---- */
.hypothesis-card {
    background: rgba(26, 35, 50, 0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.hypothesis-card .type-badge {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(26, 35, 50, 0.4);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #10b981 !important;
}

/* ---- Input fields ---- */
.stTextInput > div > div {
    background: rgba(26, 35, 50, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    color: #e2e8f0;
}
.stTextInput > div > div:focus-within {
    border-color: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15);
}
.stTextArea > div > div {
    background: rgba(26, 35, 50, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
}
.stSelectbox > div > div {
    background: rgba(26, 35, 50, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #047857, #059669);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    transform: translateY(-1px);
}

/* ---- Expanders ---- */
.streamlit-expanderHeader {
    background: rgba(26, 35, 50, 0.4);
    border-radius: 10px;
}

/* ---- Embedded viewers (PDF, D3 graph) ---- */
/* backdrop-filter creates new stacking contexts that can hide iframes */
iframe[title="streamlit_components.v1.components.html"] {
    position: relative !important;
    z-index: 50 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    position: relative;
    z-index: 1;
    overflow: visible;
}
.element-container:has(iframe) {
    position: relative;
    z-index: 50;
    overflow: visible;
}

/* ---- Dividers ---- */
hr {
    border-color: rgba(148, 163, 184, 0.1);
}

/* ---- Hero Section ---- */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}
</style>
"""


def inject_css(theme=None):
    """
    Inject the theme CSS into Streamlit.
    
    Args:
        theme: 'dark' (default) or 'light'. If None, reads from session_state.
    """
    import streamlit as st
    
    # Auto-detect theme from session_state if not specified
    if theme is None:
        theme = st.session_state.get("theme", "dark")
    
    if theme == "light":
        from src.ui.theme_light import LIGHT_CSS
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)
    else:
        st.markdown(MAIN_CSS, unsafe_allow_html=True)


def theme_toggle():
    """
    Render a theme toggle button and return current theme.
    
    Returns:
        Current theme: 'dark' or 'light'
    """
    import streamlit as st
    
    # Initialize theme in session state
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    
    # Toggle button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🌙" if st.session_state.theme == "dark" else "☀️", 
                     help="Toggle dark/light mode",
                     use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    
    return st.session_state.theme


def metric_card(value, label, icon=""):
    """Render a glassmorphism metric card."""
    return f"""
    <div class="metric-card">
        <div style="font-size:1.5rem;margin-bottom:0.3rem">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def result_card(title, score=None, domain=None, year=None, snippet=None):
    """Render a search result card."""
    meta_parts = []
    if score is not None:
        meta_parts.append(f"Score: {score:.3f}")
    if year:
        meta_parts.append(f"Year: {year}")
    meta = " · ".join(meta_parts)
    
    domain_html = ""
    if domain:
        domain_html = f'<span class="domain-badge">{domain}</span>'
    
    snippet_html = ""
    if snippet:
        snippet_html = f'<div class="result-snippet">{snippet[:200]}{"..." if len(snippet) > 200 else ""}</div>'
    
    return f"""
    <div class="result-card">
        <div class="result-title">{title}</div>
        <div class="result-meta">{meta} {domain_html}</div>
        {snippet_html}
    </div>
    """


def score_bar(label, value, max_val=1.0, color="#10b981"):
    """Render a horizontal score bar."""
    pct = min((value / max_val) * 100, 100) if max_val > 0 else 0
    return f"""
    <div class="score-bar-container">
        <span class="score-bar-label">{label}</span>
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{pct:.1f}%; background:{color}"></div>
        </div>
        <span class="score-bar-value">{value:.1%}</span>
    </div>
    """


def hypothesis_card(statement, hyp_type, confidence, confidence_score, rationale="", experiments=None):
    """Render a hypothesis card."""
    exp_html = ""
    if experiments:
        exp_items = "".join(f"<li>{e}</li>" for e in experiments[:3])
        exp_html = f'<div style="margin-top:0.5rem;font-size:0.85rem;color:#94a3b8"><b>Experiments:</b><ul style="margin:0.3rem 0">{exp_items}</ul></div>'
    
    return f"""
    <div class="hypothesis-card">
        <span class="type-badge">{hyp_type}</span>
        <div style="color:#e2e8f0;font-size:0.95rem;margin:0.5rem 0;line-height:1.5">{statement}</div>
        <div style="color:#64748b;font-size:0.85rem"><b>Confidence:</b> {confidence} ({confidence_score:.0%})</div>
        <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem">{rationale}</div>
        {exp_html}
    </div>
    """


def status_indicator(label, online=True):
    """Render a service status indicator."""
    cls = "online" if online else "offline"
    return f'<span class="status-dot {cls}"></span>{label}'
