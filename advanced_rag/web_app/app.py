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

logger = logging.getLogger(__name__)


def main():
    # Load .env if present, then fall back to Streamlit secrets.
    load_dotenv()
    try:
        if not os.getenv("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass

    st.set_page_config(page_title="VolveRAG", layout="wide")
    st.title("VolveRAG")
    st.markdown("This application lets you query the Volve petrophysics reports using natural language. Ask questions about wells, formations, petrophysical parameters, and more.")
    st.markdown("**Example:** *\"What is the water saturation value of Hugin formation in 15/9-F-5?\"*")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to your environment/.env and restart.")
        st.stop()

    with st.sidebar:
        st.header("Config")
        # Default persist dir: resolve relative to this module so it works
        # both when running from `advanced_rag/` and when Streamlit runs the
        # module from the repo root (Streamlit Cloud clones to /mount/src/...).
        default_persist = str(Path(__file__).resolve().parents[1] / "data" / "vectorstore")
        persist_dir = st.text_input("Vectorstore dir", value=default_persist)
        embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
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

    # Chat history (multi-turn)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "viewer" not in st.session_state:
        st.session_state.viewer = {"path": None, "page": None}

    col_left, col_right = st.columns([0.55, 0.45], gap="large")

    with col_left:
        st.subheader("Chat")
        st.caption("Context is preserved in this chat: you can answer clarifications (e.g., just \"matrix density\") and the prior well/formation will be remembered.")

        # Render chat history
        for msg_idx, m in enumerate(st.session_state.messages):
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            else:
                with st.chat_message("assistant"):
                    ans_body = re.sub(r"(?m)^\s*Source:\s*.*\n?", "", content).strip() if isinstance(content, str) else content
                    st.write(ans_body)

                    # Citations + viewer hook (first citation only)
                    cits = _parse_citations(content if isinstance(content, str) else "")
                    if cits:
                        c = cits[0]
                        page = c.page_start
                        cols = st.columns([0.72, 0.28])
                        with cols[0]:
                            # Clean up the source path for display
                            clean_path = _clean_source_path(c.source_path)
                            st.caption(
                                f"`{clean_path}`"
                                + (f" (pages {c.page_start}-{c.page_end})" if c.page_start else "")
                            )
                        with cols[1]:
                            label = f"View page {page}" if page else "View"
                            # Add idx component to avoid duplicate Streamlit keys when the same
                            # citation repeats across reruns or messages.
                            key_suffix = f"{len(st.session_state.messages)}_{msg_idx}_{0}_{c.source_path}_{page}"
                            if st.button(label, key=f"view_{key_suffix}"):
                                st.session_state.viewer = {"path": c.source_path, "page": page}

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
            with st.spinner("Thinking..."):
                result = graph.invoke({"messages": st.session_state.messages})
                answer = result["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Debug information expander
                with st.expander("🔍 Debug Info (click to view diagnostics)", expanded=False):
                    st.write("**Query:**", user_input)
                    
                    # Check if petro cache exists
                    vectorstore_dir = Path(__file__).resolve().parents[1] / "data" / "vectorstore"
                    cache_path = vectorstore_dir / "petro_params_cache.json"
                    cache_exists = cache_path.exists()
                    st.write(f"**Petro Cache Exists:** {cache_exists}")
                    if cache_exists:
                        st.write(f"**Cache Path:** `{cache_path}`")
                    else:
                        st.write(f"**Cache Path (not found):** `{cache_path}`")
                    
                    # Show normalized query info
                    try:
                        from src.normalize.query_normalizer import normalize_query, extract_well
                        nq = normalize_query(user_input)
                        extracted_well = extract_well(user_input)
                        st.write(f"**Extracted Well:** `{extracted_well}`")
                        st.write(f"**Normalized Query Well:** `{nq.well}`")
                        st.write(f"**Normalized Query Formation:** `{nq.formation}`")
                        st.write(f"**Normalized Query Property:** `{nq.property}`")
                    except Exception as e:
                        st.write(f"**Error extracting well:** {e}")
                    
                    # Check routing conditions
                    ql = user_input.lower()
                    param_keywords = ["petrophysical parameters", "petrophysical parameter", "net to gross", "net-to-gross", "netgros", "net/gross", "ntg", "n/g", "phif", "phi", "poro", "porosity", "water saturation", "sw", "klogh", "permeability", "permeab", "perm"]
                    has_param_keyword = any(k in ql for k in param_keywords) or bool(re.search(r'\bsw\b', ql, re.IGNORECASE))
                    extracted_well_for_routing = extracted_well if 'extracted_well' in locals() else None
                    has_well_pattern = ("15" in ql and "9" in ql) or (extracted_well_for_routing is not None)
                    should_route = has_param_keyword and (cache_exists or has_well_pattern)
                    
                    st.write("---")
                    st.write("**Routing Analysis:**")
                    st.write(f"- Has Param Keyword: `{has_param_keyword}`")
                    st.write(f"- Has Well Pattern: `{has_well_pattern}`")
                    st.write(f"- Should Route to Petro Params: `{should_route}`")
                    
                    # Check what tools were called
                    tool_calls = []
                    for msg in result.get("messages", []):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls.append(tc.get("name", "unknown"))
                    if tool_calls:
                        st.write(f"**Tools Called:** {', '.join(tool_calls)}")
                    else:
                        st.write("**Tools Called:** None (query went to retriever/LLM)")
                    
                    # Show answer source
                    if "[PETRO_PARAMS_JSON]" in answer or "[PETRO_PARAMS]" in answer:
                        st.write("**Answer Source:** Petro Params Tool (structured)")
                    elif "Source:" in answer:
                        st.write("**Answer Source:** Retriever Tool (vector search)")
                    else:
                        st.write("**Answer Source:** LLM (generated)")
                
                st.rerun()

    with col_right:
        st.subheader("PDF Viewer")
        vp = st.session_state.viewer.get("path")
        vpage = st.session_state.viewer.get("page")
        if vp:
            # Show full PDF viewer with navigation (starts at cited page)
            initial_page = vpage if isinstance(vpage, int) and vpage > 0 else 1
            components.html(_pdf_full_viewer(vp, initial_page), height=900)
        else:
            st.info("Click **View page** next to a source to open it here.")


if __name__ == "__main__":
    main()

