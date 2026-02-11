"""
Light mode theme CSS for EcoloGRAPH.

Provides a clean, professional light theme as alternative to dark mode.
"""

LIGHT_CSS = """
<style>
/* ---- Global Light Mode ---- */
.stApp {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%);
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar Light Mode ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    border-right: 1px solid #dee2e6;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #059669;
}

/* ---- Cards Light Mode ---- */
.glass-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid #dee2e6;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.glass-card:hover {
    border-color: #10b981;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    transform: translateY(-2px);
}

/* ---- Metric Cards Light Mode ---- */
.metric-card {
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #dee2e6;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.metric-card:hover {
    border-color: #10b981;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #059669;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.85rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

/* ---- Result Cards Light Mode ---- */
.result-card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(8px);
    border: 1px solid #e9ecef;
    border-left: 3px solid #10b981;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.25s ease;
}
.result-card:hover {
    background: rgba(255, 255, 255, 1);
    border-left-color: #059669;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1);
}
.result-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #212529;
    margin-bottom: 0.3rem;
}
.result-meta {
    font-size: 0.8rem;
    color: #6c757d;
}
.result-snippet {
    font-size: 0.9rem;
    color: #495057;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* ---- Domain Badge Light Mode ---- */
.domain-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}

/* ---- Input fields Light Mode ---- */
.stTextInput > div > div {
    background: white;
    border: 1px solid #ced4da;
    border-radius: 12px;
    color: #212529;
}
.stTextInput > div > div:focus-within {
    border-color: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}
.stTextArea > div > div {
    background: white;
    border: 1px solid #ced4da;
    border-radius: 12px;
}
.stSelectbox > div > div {
    background: white;
    border: 1px solid #ced4da;
    border-radius: 12px;
}

/* ---- Buttons Light Mode ---- */
.stButton > button {
    background: linear-gradient(135deg, #10b981, #34d399);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #059669, #10b981);
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
    transform: translateY(-1px);
}

/* ---- Expanders Light Mode ---- */
.streamlit-expanderHeader {
    background: rgba(248, 249, 250, 0.8);
    border-radius: 10px;
}

/* ---- Dividers Light Mode ---- */
hr {
    border-color: #dee2e6;
}

/* ---- Hero Section Light Mode ---- */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #059669, #10b981, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1rem;
    color: #6c757d;
    margin-bottom: 2rem;
}

/* Override text colors for light mode */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: #212529 !important;
}
.stMarkdown {
    color: #212529;
}
</style>
"""
