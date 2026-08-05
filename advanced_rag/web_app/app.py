"""
Streamlit Web UI for the LangGraph-based Petrophysical RAG system.

Goal: click a source and open the exact cited page inside the app (embedded viewer).

Notes:
- For deterministic tools ([SECTION] / [PETRO_PARAMS]) we already emit `Source: ... (pages a-b)` lines.
- This UI parses those citations and embeds the PDF with a best-effort `#page=` fragment.
"""

from __future__ import annotations

import os
import re
import uuid
import logging
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components

# Configure logging to ensure messages are visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing config
)

# Import from extracted modules
from .logic.citation_parser import Citation, _parse_citations, _clean_source_path
from .logic.asset_downloader import _ensure_pdfs_available, _ensure_vectorstore_available
from .logic.pdf_viewer import _pdf_full_viewer
from .logic.graph_manager import _get_graph

# Import modern UI components
from .components import (
    render_header,
    render_chat_message,
    render_citation_card,
    render_sota_panel,
    render_pdf_viewer_placeholder,
    render_loading_spinner,
    render_example_queries,
    render_debug_expander,
)

logger = logging.getLogger(__name__)


def main():
    # Load .env if present, then fall back to Streamlit secrets.
    load_dotenv()
    try:
        if not os.getenv("GROQ_API_KEY") and "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = str(st.secrets["GROQ_API_KEY"])
        if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
        if "LLM_PROVIDER" in st.secrets:
            os.environ.setdefault("LLM_PROVIDER", str(st.secrets["LLM_PROVIDER"]))
        if "EMBEDDING_PROVIDER" in st.secrets:
            os.environ.setdefault("EMBEDDING_PROVIDER", str(st.secrets["EMBEDDING_PROVIDER"]))
    except Exception:
        pass

    st.set_page_config(
        page_title="VolveRAG", 
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Render modern header
    render_header(
        title="VolveRAG",
        subtitle="Advanced Petrophysical RAG System for Volve Field Data"
    )
    
    st.markdown(
        """
        <style>
        .main-info {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin-bottom: 2rem;
        }
        </style>
        <div class="main-info">
            <strong>🎯 What can I do?</strong> Ask questions about wells, formations, petrophysical parameters, and more using natural language.
            <br><em>Example: "What is the water saturation value of Hugin formation in 15/9-F-5?"</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.getenv("LLM_PROVIDER", "groq").lower() == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Add it to Streamlit secrets or your local .env and restart.")
        st.stop()
    if os.getenv("EMBEDDING_PROVIDER", "huggingface").lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is required only when EMBEDDING_PROVIDER=openai.")
        st.stop()

    with st.sidebar:
        st.header("Config")
        # Default persist dir: resolve relative to this module so it works
        # both when running from `advanced_rag/` and when Streamlit runs the
        # module from the repo root (Streamlit Cloud clones to /mount/src/...).
        default_persist = str(Path(__file__).resolve().parents[1] / "data" / "vectorstore")
        persist_dir = st.text_input("Vectorstore dir", value=default_persist)
        embedding_model = st.text_input("Embedding model", value=os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"))
        st.caption("Index build remains CLI-based: `python -m src.main --build-index`.")
        
        # Debug: show current working directory and vectorstore path
        cwd = Path.cwd()
        vs_path = Path(persist_dir)
        vs_abs = vs_path.resolve()
        st.caption(f"🔍 CWD: {cwd}")
        st.caption(f"🔍 Vectorstore: {vs_abs}")
        st.caption(f"🔍 Exists: {vs_abs.exists()}")

        # Also print to stdout so Streamlit Cloud deployment logs capture it
        print(f"DEBUG: __file__ resolved to: {Path(__file__).resolve()}")
        print(f"DEBUG: default_persist: {default_persist}")
        print(f"DEBUG: CWD: {cwd}")
        print(f"DEBUG: Vectorstore path: {vs_abs}")
        print(f"DEBUG: Vectorstore exists: {vs_abs.exists()}")

        # ── SOTA Techniques Status ────────────────────────────────────────
        st.divider()
        
        # Use modern SOTA panel component
        render_sota_panel()
        # ─────────────────────────────────────────────────────────────────

    # Chat history (multi-turn)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "viewer" not in st.session_state:
        st.session_state.viewer = {"path": None, "page": None}

    col_left, col_right = st.columns([0.58, 0.42], gap="large")

    with col_left:
        # Render example queries if no messages yet
        if not st.session_state.messages:
            render_example_queries()
            
            # Check if user clicked an example query
            if hasattr(st.session_state, 'example_query') and st.session_state.example_query:
                user_input = st.session_state.example_query
                st.session_state.example_query = None  # Clear after use
        
        st.markdown("### 💬 Chat")
        st.caption("Context is preserved in this chat: you can answer clarifications (e.g., just \"matrix density\") and the prior well/formation will be remembered.")

        # Render chat history with modern styling
        for msg_idx, m in enumerate(st.session_state.messages):
            role = m.get("role")
            content = m.get("content", "")
            
            if role == "user":
                render_chat_message(role, content, idx=msg_idx)
            else:
                # Parse citations for assistant messages
                cits = _parse_citations(content if isinstance(content, str) else "")
                
                # Display the message content
                render_chat_message(role, content, citations=cits, idx=msg_idx)
                
                # Show citation cards if available
                if cits:
                    for cit_idx, c in enumerate(cits):
                        if c.source_path and c.source_path != "N/A":
                            key_suffix = f"{len(st.session_state.messages)}_{msg_idx}_{cit_idx}_{c.source_path}_{c.page_start}"
                            
                            def on_click(path=c.source_path, page=c.page_start):
                                st.session_state.viewer = {"path": path, "page": page}
                            
                            render_citation_card(
                                source_path=c.source_path,
                                page_start=c.page_start,
                                page_end=c.page_end,
                                on_click=on_click,
                                key=f"cite_{key_suffix}"
                            )

        # Ensure vectorstore is available (download if needed)
        if not _ensure_vectorstore_available(persist_dir):
            # Show a message if download is in progress or failed
            vectorstore_url = os.getenv("VECTORSTORE_URL")
            if not vectorstore_url:
                try:
                    if "VECTORSTORE_URL" in st.secrets:
                        vectorstore_url = str(st.secrets["VECTORSTORE_URL"])
                except Exception:
                    pass
            
            # Auto-fix: Update old repository name (VolveRAG) to new name (volverag)
            if vectorstore_url and "VolveRAG" in vectorstore_url:
                vectorstore_url = vectorstore_url.replace("VolveRAG", "volverag")
                logger.info(f"Auto-corrected repository name in URL: {vectorstore_url}")
            
            if vectorstore_url:
                with st.spinner("Downloading vectorstore... This may take a few minutes on first run."):
                    if _ensure_vectorstore_available(persist_dir):
                        st.success("Vectorstore downloaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to download vectorstore. Please check the VECTORSTORE_URL and try again.")
                        st.info(f"Attempted URL: {vectorstore_url}")
                        # Provide helpful guidance
                        st.warning("""
                        **Common issues:**
                        - Repository name changed from `VolveRAG` to `volverag` (lowercase)
                        - Make sure the release exists: https://github.com/samwiserr/volverag/releases
                        - Correct URL format: `https://github.com/samwiserr/volverag/releases/download/TAG/FILENAME.zip`
                        - Update `VECTORSTORE_URL` in Streamlit Cloud Secrets with the new repository name
                        """)
                        st.stop()
            else:
                # No URL configured - show the original error message
                st.error("⚠️ **Vector store not found!**")
                st.info("""
                **To build the index and use this application:**
                
                1. **Download the Volve dataset** from the official Equinor source
                2. **Place it outside the repository** (e.g., `../spwla_volve-main/`)
                3. **Build the index** by running:
                   ```bash
                   cd advanced_rag
                   python -m src.main --build-index --documents-path ../spwla_volve-main
                   ```
                4. **Restart this application**
                
                **Note:** The vector store must be built locally before deploying to Streamlit Cloud.
                See [DATA_POLICY.md](../DATA_POLICY.md) for details on why data files are not in the repository.
                
                **Alternative:** Set `VECTORSTORE_URL` in Streamlit Cloud Secrets to automatically download a pre-built vectorstore.
                """)
                st.stop()
                return
        
        # Ensure PDFs are available (download if needed, but don't block if unavailable)
        pdfs_dir = Path(__file__).resolve().parents[1] / "data" / "pdfs"
        if not _ensure_pdfs_available(pdfs_dir):
            # Show info in sidebar but don't block - app can work without PDFs
            pdfs_url = os.getenv("PDFS_URL")
            if not pdfs_url:
                try:
                    if "PDFS_URL" in st.secrets:
                        pdfs_url = str(st.secrets["PDFS_URL"])
                except Exception:
                    pass
            
            # Auto-fix: Update old repository name (VolveRAG) to new name (volverag)
            if pdfs_url and "VolveRAG" in pdfs_url:
                pdfs_url = pdfs_url.replace("VolveRAG", "volverag")
                logger.info(f"Auto-corrected repository name in PDFS_URL: {pdfs_url}")
            
            if pdfs_url:
                with st.sidebar:
                    with st.spinner("Downloading PDFs... This may take a few minutes."):
                        if _ensure_pdfs_available(pdfs_dir):
                            st.success("PDFs downloaded!")
                            st.rerun()
                        else:
                            st.warning("PDFs not available. PDF viewer will show messages instead of documents.")
            else:
                # No PDFS_URL configured - silently continue (PDFs optional)
                pass
        
        # Check if vectorstore exists before allowing queries
        graph = _get_graph(persist_dir, embedding_model, cache_version=2)
        if graph is None:
            st.error("⚠️ **Vector store not found!**")
            st.info("""
            **To build the index and use this application:**
            
            1. **Download the Volve dataset** from the official Equinor source
            2. **Place it outside the repository** (e.g., `../spwla_volve-main/`)
            3. **Build the index** by running:
               ```bash
               cd advanced_rag
               python -m src.main --build-index --documents-path ../spwla_volve-main
               ```
            4. **Restart this application**
            
            **Note:** The vector store must be built locally before deploying to Streamlit Cloud.
            See [DATA_POLICY.md](../DATA_POLICY.md) for details on why data files are not in the repository.
            """)
            st.stop()
            return
        
        # Chat input
        user_input = st.chat_input("Ask a question (typos ok).")
        if user_input and user_input.strip():
            # Input validation and sanitization
            from src.core.validation import validate_query
            from src.core.security import sanitize_input, get_rate_limiter
            
            # Validate query
            is_valid, error_msg = validate_query(user_input)
            if not is_valid:
                st.error(f"⚠️ **Invalid query:** {error_msg}")
                st.info("Please ensure your query is between 1 and 2000 characters and doesn't contain dangerous patterns.")
                return
            
            # Sanitize input
            sanitize_result = sanitize_input(user_input)
            if sanitize_result.is_err():
                st.error(f"⚠️ **Input sanitization failed:** {sanitize_result.error().message}")
                return
            
            sanitized_input = sanitize_result.unwrap()
            
            # Rate limiting (use session ID as identifier)
            session_id = st.session_state.get("session_id", "default")
            if "session_id" not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
                session_id = st.session_state.session_id
            
            rate_limiter = get_rate_limiter()
            rate_check = rate_limiter.check_rate_limit(session_id)
            if rate_check.is_err():
                error = rate_check.error()
                st.warning(f"⚠️ **Rate limit exceeded:** {error.message}")
                remaining = rate_limiter.get_remaining(session_id)
                st.info(f"You have {remaining} requests remaining. Please wait a moment before trying again.")
                return
            
            st.session_state.messages.append({"role": "user", "content": sanitized_input})
            
            # Show modern loading spinner
            render_loading_spinner("Analyzing your query...")
            loading_placeholder = st.empty()
            
            with loading_placeholder:
                result = graph.invoke({"messages": st.session_state.messages})
                answer = result["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Use modern debug expander
                render_debug_expander(result, user_input)
                
                st.rerun()

    with col_right:
        st.markdown("### 📄 PDF Viewer")
        vp = st.session_state.viewer.get("path")
        vpage = st.session_state.viewer.get("page")
        if vp:
            # Show full PDF viewer with navigation (starts at cited page)
            initial_page = vpage if isinstance(vpage, int) and vpage > 0 else 1
            components.html(_pdf_full_viewer(vp, initial_page), height=900)
        else:
            render_pdf_viewer_placeholder()


if __name__ == "__main__":
    main()

