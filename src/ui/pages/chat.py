"""
EcoloGRAPH — Chat Page.

Interactive chat with the LangGraph agent.
Supports tool calls, streaming responses, and conversation history.
"""
import streamlit as st
from src.ui.theme import inject_css


def render():
    inject_css()

    st.markdown(
        '<div class="hero-title">💬 EcoloGRAPH Chat</div>'
        '<div class="hero-subtitle">Ask questions about your indexed papers, species, and ecological data</div>',
        unsafe_allow_html=True,
    )

    # --- Initialize agent (cached) ---
    agent = _get_agent()
    if agent is None:
        st.error("⚠️ Could not connect to the LLM. Make sure Ollama is running.")
        st.code("# Start Ollama and pull a model:\nollama pull qwen3:8b", language="bash")
        return

    # Show agent info
    try:
        info = agent.get_info()
        cols = st.columns(3)
        cols[0].markdown(
            f'<div class="glass-card"><span style="color:#10b981">🤖 Model:</span> '
            f'<strong style="color:#e2e8f0">{info["model"]}</strong></div>',
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f'<div class="glass-card"><span style="color:#3b82f6">🔧 Tools:</span> '
            f'<strong style="color:#e2e8f0">{len(info["tools"])}</strong></div>',
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            f'<div class="glass-card"><span style="color:#f59e0b">📚 Available:</span> '
            f'<strong style="color:#e2e8f0">{", ".join(info["tools"])}</strong></div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.markdown("---")

    # --- Conversation history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍🔬"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🌿"):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    with st.expander("🔧 Tool calls", expanded=False):
                        for tc in msg["tool_calls"]:
                            st.markdown(f"- {tc}")

    # --- Chat input ---
    if prompt := st.chat_input("Ask about species, papers, domains..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant", avatar="🌿"):
            tool_calls = []
            answer_text = ""

            with st.spinner("🧠 Thinking..."):
                try:
                    status_placeholder = st.empty()
                    
                    for event_type, content in agent.ask_streaming(prompt, history=st.session_state.messages[:-1]):
                        if event_type == "routing":
                            status_placeholder.markdown(
                                f'<div style="color:#64748b;font-size:0.85rem">🧭 {content}</div>',
                                unsafe_allow_html=True,
                            )
                        elif event_type == "tool_call":
                            tool_calls.append(content)
                            status_placeholder.markdown(
                                f'<div style="color:#3b82f6;font-size:0.85rem">🔧 {content}</div>',
                                unsafe_allow_html=True,
                            )
                        elif event_type == "tool_result":
                            preview = content[:150].replace("\n", " ")
                            status_placeholder.markdown(
                                f'<div style="color:#64748b;font-size:0.85rem">📋 {preview}...</div>',
                                unsafe_allow_html=True,
                            )
                        elif event_type == "answer":
                            answer_text = content
                        elif event_type == "error":
                            answer_text = f"❌ {content}"

                    status_placeholder.empty()

                except Exception as e:
                    answer_text = f"❌ Error: {str(e)}"

            # Display answer
            if answer_text:
                st.markdown(answer_text)
            else:
                answer_text = "No response received."
                st.warning(answer_text)

            # Show tool calls in expander
            if tool_calls:
                with st.expander("🔧 Tool calls used", expanded=False):
                    for tc in tool_calls:
                        st.markdown(f"- {tc}")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "tool_calls": tool_calls,
            })

    # --- Sidebar ---
    with st.sidebar:
        # Chat history export
        st.markdown("---")
        st.markdown("#### 💾 Chat History")

        if st.session_state.get("messages"):
            col_md, col_txt = st.columns(2)
            with col_md:
                if st.button("📥 Export .md", use_container_width=True):
                    _export_chat("md")
            with col_txt:
                if st.button("📥 Export .txt", use_container_width=True):
                    _export_chat("txt")
        else:
            st.caption("Start a conversation to enable export.")

        # Browse past conversations
        from pathlib import Path
        history_dir = Path("data/chat_history")
        if history_dir.exists():
            files = sorted(history_dir.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                with st.expander(f"📂 History ({len(files)} chats)", expanded=False):
                    for f in files[:20]:
                        col_f, col_load = st.columns([3, 1])
                        col_f.caption(f.name)
                        if col_load.button("📖", key=f"load_{f.name}"):
                            st.session_state["_view_history"] = f.read_text(encoding="utf-8")

        # Show history viewer
        if "_view_history" in st.session_state:
            st.markdown("---")
            st.markdown("#### 📜 Past Conversation")
            st.markdown(st.session_state["_view_history"])
            if st.button("✖ Close", key="close_history"):
                del st.session_state["_view_history"]
                st.rerun()

        # Quick examples
        st.markdown("---")
        st.markdown("#### 💡 Try asking:")
        examples = [
            "What species are most studied in coral reef ecology?",
            "Tell me about Gadus morhua",
            "What domains are related to conservation?",
            "Generate a hypothesis about ocean acidification",
            "Search papers about microplastics",
            "What tools do you have?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state["_pending_prompt"] = ex
                st.rerun()

    # Handle example button clicks
    if "_pending_prompt" in st.session_state:
        pending = st.session_state.pop("_pending_prompt")
        st.session_state.messages.append({"role": "user", "content": pending})
        st.rerun()


def _export_chat(fmt: str):
    """Export current conversation to data/chat_history/."""
    from pathlib import Path
    from datetime import datetime

    history_dir = Path("data/chat_history")
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp}.{fmt}"

    lines = []
    for msg in st.session_state.get("messages", []):
        role_label = "🧑‍🔬 **User**" if msg["role"] == "user" else "🌿 **EcoloGRAPH**"
        if fmt == "md":
            lines.append(f"### {role_label}\n\n{msg['content']}\n")
            if msg.get("tool_calls"):
                tools_str = ", ".join(str(tc) for tc in msg["tool_calls"])
                lines.append(f"**Tools used:** {tools_str}\n")
        else:
            role_tag = msg["role"].upper()
            lines.append(f"[{role_tag}]\n{msg['content']}\n")
            if msg.get("tool_calls"):
                lines.append(f"[TOOLS] {', '.join(str(tc) for tc in msg['tool_calls'])}\n")

    filepath = history_dir / filename
    filepath.write_text("\n".join(lines), encoding="utf-8")
    st.success(f"✅ Saved: {filename}")


@st.cache_resource
def _get_agent():
    """Initialize the QueryAgent (cached across reruns)."""
    try:
        from src.core.config import _load_api_key_file, get_settings
        _load_api_key_file()

        settings = get_settings()
        base_url = settings.llm.base_url
        model = settings.llm.reasoning_model  # Use reasoning model for chat/agent

        import os
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")

        from src.agent import QueryAgent
        agent = QueryAgent(model=model, api_key=api_key, base_url=base_url)
        # Verify connection
        agent.get_info()
        return agent
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Agent init failed: {e}")
        return None
