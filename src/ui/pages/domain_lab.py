"""
EcoloGRAPH — Domain Classifier Page.

Scientific text classification using the 43-domain classifier.
"""
import streamlit as st
from src.ui.theme import inject_css, score_bar


def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">🔬 Domain Classifier</div>'
        '<div class="hero-subtitle">Classify scientific text into ecological domains</div>',
        unsafe_allow_html=True,
    )

    _render_classify()


def _render_classify():
    text = st.text_area(
        "Paste text to classify",
        placeholder="Paste an abstract, paragraph, or any scientific text here...",
        height=180,
        key="classify_input",
    )

    if st.button("🏷️ Classify", key="classify_btn") and text:
        with st.spinner("Classifying..."):
            try:
                from src.extraction.domain_classifier import DomainClassifier
                classifier = DomainClassifier()
                result = classifier.classify_text(text, use_llm=False)

                # Primary domain
                st.markdown(
                    f'<div class="glass-card">'
                    f'<div style="font-size:0.8rem;color:#64748b;text-transform:uppercase">Primary Domain</div>'
                    f'<div style="font-size:1.5rem;font-weight:700;color:#10b981">'
                    f'{result.primary_domain.value.replace("_", " ").title()}'
                    f'</div>'
                    f'<div style="color:#94a3b8">Confidence: {result.confidence:.1%} · Method: {result.method}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Study type
                if result.study_type.value != "unknown":
                    st.markdown(f"**Study type:** {result.study_type.value.replace('_', ' ').title()} "
                                f"({result.study_type_confidence:.0%})")

                # Domain scores
                st.markdown("#### Domain Scores")
                top_domains = classifier.get_top_domains(result, threshold=0.03)

                colors = [
                    "#10b981", "#34d399", "#3b82f6", "#8b5cf6",
                    "#f59e0b", "#ef4444", "#ec4899", "#06b6d4",
                ]

                bars_html = ""
                for i, (dt, sc) in enumerate(top_domains[:10]):
                    color = colors[i % len(colors)]
                    bars_html += score_bar(
                        dt.value.replace("_", " ").title(),
                        sc,
                        color=color,
                    )

                st.markdown(bars_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Classification error: {e}")

    elif not text:
        st.markdown(
            '<div class="glass-card" style="text-align:center;padding:2rem">'
            '<div style="font-size:2.5rem">🏷️</div>'
            '<div style="color:#94a3b8;margin-top:0.5rem">Paste scientific text to classify it into domains</div>'
            '</div>',
            unsafe_allow_html=True,
        )
