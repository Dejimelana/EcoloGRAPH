"""
EcoloGRAPH — Papers Explorer Page.

Browse, filter, and inspect indexed papers.
Includes PDF viewer and extracted entity display.
"""
import base64
import streamlit as st
from pathlib import Path
from src.ui.theme import inject_css


def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">📄 Papers Explorer</div>'
        '<div class="hero-subtitle">Browse, filter, and inspect your indexed research papers</div>',
        unsafe_allow_html=True,
    )

    # --- Load papers ---
    try:
        from src.search.paper_index import PaperIndex, SearchFilters
        idx = PaperIndex()
        total = idx.count()
    except Exception as e:
        st.error(f"Could not connect to paper index: {e}")
        return

    if total == 0:
        st.info(
            "No papers indexed yet. Run the ingestion pipeline first:\n\n"
            "```\npython scripts/ingest.py data/raw/\n```"
        )
        return

    # --- Filters sidebar ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### 🔍 Filter Papers")

        # Domain filter
        domains = idx.get_domains()
        domain_names = ["All"] + [d[0].replace("_", " ").title() for d in domains]
        domain_raw = ["All"] + [d[0] for d in domains]
        selected_domain_idx = st.selectbox(
            "Domain",
            range(len(domain_names)),
            format_func=lambda i: f"{domain_names[i]} ({domains[i-1][1]})" if i > 0 else "All",
        )
        selected_domain = None if selected_domain_idx == 0 else domain_raw[selected_domain_idx]

        # Year filter
        year_range = idx.get_year_range()
        if year_range[0] and year_range[1] and year_range[0] != year_range[1]:
            year_min, year_max = st.slider(
                "Year range",
                min_value=year_range[0],
                max_value=year_range[1],
                value=(year_range[0], year_range[1]),
            )
        else:
            year_min, year_max = year_range

        # Keyword search
        keyword = st.text_input("Keyword search", placeholder="e.g. coral, fish, climate")

    # --- Build filters ---
    filters = SearchFilters(
        domains=[selected_domain] if selected_domain else None,
        year_min=year_min if year_min != year_range[0] else None,
        year_max=year_max if year_max != year_range[1] else None,
    )

    # --- Get papers ---
    if keyword:
        results = idx.search(keyword, filters=filters, limit=100)
        papers_data = []
        for r in results:
            p = idx.get_paper(r.doc_id)
            if p:
                papers_data.append(p)
    else:
        papers_data = idx.get_all_papers(limit=200)
        # Apply domain filter manually for get_all_papers
        if selected_domain:
            papers_data = [p for p in papers_data if p.primary_domain == selected_domain]
        if filters.year_min:
            papers_data = [p for p in papers_data if p.year and p.year >= filters.year_min]
        if filters.year_max:
            papers_data = [p for p in papers_data if p.year and p.year <= filters.year_max]

    # --- Summary bar ---
    st.markdown(
        f'<div class="glass-card" style="padding:0.8rem 1.2rem">'
        f'<span style="color:#10b981;font-weight:600">{len(papers_data)}</span>'
        f'<span style="color:#94a3b8"> papers'
        f'{" in " + selected_domain.replace("_", " ").title() if selected_domain else ""}'
        f'{" matching "" + keyword + """ if keyword else ""}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # --- Paper detail view ---
    if "selected_paper_id" in st.session_state and st.session_state.selected_paper_id:
        _render_paper_detail(idx, st.session_state.selected_paper_id)
        if st.button("← Back to paper list", key="back_btn"):
            st.session_state.selected_paper_id = None
            st.rerun()
        return

    # --- Paper list ---
    if not papers_data:
        st.warning("No papers match your filters.")
        return

    for i, paper in enumerate(papers_data):
        _render_paper_card(paper, i)


def _render_paper_card(paper, index):
    """Render a clickable paper card."""
    domain_label = (paper.primary_domain or "unknown").replace("_", " ").title()
    authors = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors += f" +{len(paper.authors) - 3} more"

    year_str = str(paper.year) if paper.year else "N/A"

    col_main, col_action = st.columns([5, 1])

    with col_main:
        st.markdown(
            f'<div class="glass-card" style="cursor:pointer;margin-bottom:0.5rem">'
            f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.3rem">'
            f'<span class="domain-badge">{domain_label}</span>'
            f'<span style="color:#64748b;font-size:0.85rem">{year_str}</span>'
            f'<span style="color:#64748b;font-size:0.85rem">{paper.journal or ""}</span>'
            f'</div>'
            f'<div style="font-size:1.05rem;font-weight:600;color:#e2e8f0;margin-bottom:0.2rem">'
            f'{paper.title or "Untitled"}</div>'
            f'<div style="color:#94a3b8;font-size:0.85rem">{authors}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_action:
        if st.button("📖 View", key=f"view_{index}", use_container_width=True):
            st.session_state.selected_paper_id = paper.doc_id
            st.rerun()


def _render_paper_detail(idx, doc_id):
    """Render full paper detail view with PDF viewer."""
    paper = idx.get_paper(doc_id)
    if not paper:
        st.error("Paper not found.")
        return

    # --- Header ---
    domain_label = (paper.primary_domain or "unknown").replace("_", " ").title()
    st.markdown(
        f'<div class="glass-card">'
        f'<span class="domain-badge" style="margin-bottom:0.5rem">{domain_label}</span>'
        f'<div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;margin:0.5rem 0">'
        f'{paper.title or "Untitled"}</div>'
        f'<div style="color:#94a3b8">{", ".join(paper.authors)}</div>'
        f'<div style="color:#64748b;font-size:0.9rem;margin-top:0.3rem">'
        f'{paper.journal or ""} · {paper.year or "N/A"} · ID: {paper.doc_id[:12]}...'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # --- Tabs ---
    tab_meta, tab_domains, tab_pdf = st.tabs([
        "📋 Metadata & Abstract",
        "🏷️ Domain Classification",
        "📄 PDF Viewer"
    ])

    with tab_meta:
        _render_metadata_tab(paper)

    with tab_domains:
        _render_domains_tab(paper)

    with tab_pdf:
        _render_pdf_tab(paper)


def _render_metadata_tab(paper):
    """Render metadata and abstract."""
    # Key metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Year", paper.year or "N/A")
    c2.metric("Study Type", (paper.study_type or "N/A").replace("_", " ").title())
    c3.metric("Keywords", len(paper.keywords))

    # Abstract
    if paper.abstract:
        st.markdown("### Abstract")
        st.markdown(
            f'<div class="glass-card" style="color:#cbd5e1;line-height:1.7">'
            f'{paper.abstract}</div>',
            unsafe_allow_html=True,
        )

    # Keywords
    if paper.keywords:
        st.markdown("### Keywords")
        badges = "".join(
            f'<span class="domain-badge secondary">{kw}</span>' for kw in paper.keywords
        )
        st.markdown(badges, unsafe_allow_html=True)

    # Source info
    if paper.source_path:
        st.markdown("### Source File")
        st.code(paper.source_path, language=None)


def _render_domains_tab(paper):
    """Render domain classification scores."""
    if not paper.domains:
        st.info("No domain classification data available for this paper.")
        return

    st.markdown("### Domain Affinity Scores")
    st.caption("How strongly this paper relates to each scientific domain (0–1 scale)")

    import pandas as pd

    # Sort by score descending
    sorted_domains = sorted(paper.domains.items(), key=lambda x: -x[1])

    # Top domains bar chart
    if sorted_domains:
        top = sorted_domains[:10]
        df = pd.DataFrame(
            [(d.replace("_", " ").title(), s) for d, s in top],
            columns=["Domain", "Score"],
        )
        st.bar_chart(df.set_index("Domain"), color="#10b981", height=300)

    # Full table
    st.markdown("### All Classified Domains")
    all_df = pd.DataFrame(
        [(d.replace("_", " ").title(), f"{s:.3f}", "🟢" if s > 0.3 else "🟡" if s > 0.1 else "⚪")
         for d, s in sorted_domains],
        columns=["Domain", "Score", "Strength"],
    )
    st.dataframe(all_df, use_container_width=True, hide_index=True)


def _render_pdf_tab(paper):
    """Render PDF viewer if the source file exists."""
    if not paper.source_path:
        st.warning("No source file path recorded for this paper.")
        return

    pdf_path = Path(paper.source_path)
    project_root = Path(__file__).parent.parent.parent.parent  # EcoloGRAPH root

    # Try multiple path resolution strategies
    candidates = [
        pdf_path,                                                    # as-is
        (project_root / pdf_path).resolve(),                         # relative to project
        Path(pdf_path).resolve(),                                    # relative to CWD
        project_root / "data" / "raw" / pdf_path.name,               # name in data/raw/
        project_root.parent / "scientific-rag-assistant" / "data" / "raw" / pdf_path.name,
    ]

    found_path = None
    for p in candidates:
        try:
            if p.exists():
                found_path = p
                break
        except Exception:
            continue

    if not found_path:
        st.warning(
            f"PDF file not found at: `{paper.source_path}`\n\n"
            "Run `python scripts/fix_paper_metadata.py` to resolve paths, "
            "or place PDFs in `data/raw/`."
        )
        st.info(f"**Stored path:** `{paper.source_path}`")
        return

    # Read and display PDF using PDF.js (works in sandboxed iframes)
    try:
        import streamlit.components.v1 as components

        pdf_bytes = found_path.read_bytes()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        # PDF.js renders PDFs as canvas — no browser plugin dependency
        pdf_html = '''
        <!DOCTYPE html>
        <html>
        <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { background: #0f172a; overflow-y: auto; font-family: Inter, sans-serif; }
            #controls {
                position: sticky; top: 0; z-index: 10;
                background: #1e293b; padding: 8px 16px;
                display: flex; align-items: center; gap: 12px;
                border-bottom: 1px solid #334155;
            }
            #controls button {
                background: #334155; color: #e2e8f0; border: none;
                border-radius: 6px; padding: 6px 14px; cursor: pointer;
                font-size: 13px; transition: background 0.2s;
            }
            #controls button:hover { background: #475569; }
            #controls span { color: #94a3b8; font-size: 13px; }
            #viewer { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 12px 0; }
            #viewer canvas {
                box-shadow: 0 2px 12px rgba(0,0,0,0.4);
                border-radius: 4px;
                max-width: 100%;
            }
        </style>
        </head>
        <body>
        <div id="controls">
            <button onclick="prevPage()">◀ Prev</button>
            <span id="pageInfo">Loading...</span>
            <button onclick="nextPage()">Next ▶</button>
            <button onclick="zoomOut()">−</button>
            <span id="zoomInfo">100%</span>
            <button onclick="zoomIn()">+</button>
        </div>
        <div id="viewer"></div>
        <script>
        const BASE64 = "''' + b64_pdf + '''";
        const raw = atob(BASE64);
        const uint8 = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) uint8[i] = raw.charCodeAt(i);

        pdfjsLib.GlobalWorkerOptions.workerSrc =
            "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

        let pdfDoc = null, currentPage = 1, scale = 1.2;

        async function loadPDF() {
            pdfDoc = await pdfjsLib.getDocument({data: uint8}).promise;
            document.getElementById("pageInfo").textContent =
                `Page ${currentPage} / ${pdfDoc.numPages}`;
            renderPage(currentPage);
        }

        async function renderPage(num) {
            const page = await pdfDoc.getPage(num);
            const viewport = page.getViewport({scale: scale});
            const viewer = document.getElementById("viewer");
            viewer.innerHTML = "";
            const canvas = document.createElement("canvas");
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            viewer.appendChild(canvas);
            await page.render({canvasContext: canvas.getContext("2d"), viewport}).promise;
            document.getElementById("pageInfo").textContent =
                `Page ${num} / ${pdfDoc.numPages}`;
            document.getElementById("zoomInfo").textContent =
                `${Math.round(scale * 100)}%`;
        }

        function prevPage() { if (currentPage > 1) renderPage(--currentPage); }
        function nextPage() { if (pdfDoc && currentPage < pdfDoc.numPages) renderPage(++currentPage); }
        function zoomIn() { scale = Math.min(3, scale + 0.2); renderPage(currentPage); }
        function zoomOut() { scale = Math.max(0.5, scale - 0.2); renderPage(currentPage); }

        loadPDF();
        </script>
        </body>
        </html>
        '''
        components.html(pdf_html, height=800, scrolling=True)

        # Download button
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=found_path.name,
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"Error rendering PDF: {e}")
