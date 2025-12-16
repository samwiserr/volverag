"""
Streamlit Web UI for the LangGraph-based Petrophysical RAG system.

Goal: click a source and open the exact cited page inside the app (embedded viewer).

Notes:
- For deterministic tools ([SECTION] / [PETRO_PARAMS]) we already emit `Source: ... (pages a-b)` lines.
- This UI parses those citations and embeds the PDF with a best-effort `#page=` fragment.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from src.graph.rag_graph import build_rag_graph
from src.tools.formation_properties_tool import FormationPropertiesTool
from src.tools.eval_params_tool import EvalParamsTool
from src.tools.petro_params_tool import PetroParamsTool
from src.tools.retriever_tool import RetrieverTool
from src.tools.section_lookup_tool import SectionLookupTool
from src.tools.structured_facts_tool import StructuredFactsTool
from src.tools.well_picks_tool import WellPicksTool


@dataclass
class Citation:
    source_path: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None


def _parse_citations(answer: str) -> List[Citation]:
    """
    Extract citations from tool outputs like:
      Source: C:\\path\\file.pdf (pages 12-13)
      Source: C:\\path\\file.pdf (page 12)
    """
    if not isinstance(answer, str) or not answer.strip():
        return []

    cits: List[Citation] = []
    seen = set()  # Avoid duplicates

    # Pattern 1: Source: path (pages X-Y)
    for m in re.finditer(
        r"^Source:\s*(.+?)\s*\(pages\s+(\d+)\s*-\s*(\d+)\)\s*$",
        answer,
        flags=re.MULTILINE,
    ):
        source = m.group(1).strip()
        page_start = int(m.group(2))
        page_end = int(m.group(3))
        key = (source, page_start, page_end)
        if key not in seen:
            seen.add(key)
            cits.append(Citation(source, page_start, page_end))

    # Pattern 2: Source: path (page X) - Phase 1.5 enhancement
    for m in re.finditer(
        r"^Source:\s*(.+?)\s*\(page\s+(\d+)\)\s*$",
        answer,
        flags=re.MULTILINE,
    ):
        source = m.group(1).strip()
        page = int(m.group(2))
        key = (source, page, page)
        if key not in seen:
            seen.add(key)
            cits.append(Citation(source, page, page))

    if cits:
        return cits

    # Fallback: Source: <path> (no page info)
    for m in re.finditer(r"^Source:\s*(.+?)\s*$", answer, flags=re.MULTILINE):
        source = m.group(1).strip()
        # Skip if it looks like it has page info but didn't match (avoid duplicates)
        if "(page" not in source and "(pages" not in source:
            if source not in seen:
                seen.add(source)
                cits.append(Citation(source))

    return cits


def _pdf_iframe(file_path: str, page: Optional[int]) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"<div>File not found: {p}</div>"

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    # Best-effort: encourage the viewer to land at the top of the cited page
    # (reduces cases where continuous-scroll highlights the next thumbnail).
    frag = f"#page={page}&view=FitH&zoom=page-width" if page else ""
    return (
        f'<iframe src="data:application/pdf;base64,{b64}{frag}" '
        f'width="100%" height="900" type="application/pdf"></iframe>'
    )


@st.cache_data(show_spinner=False)
def _render_pdf_page_png(file_path: str, page_1based: int, zoom: float = 2.0) -> Optional[bytes]:
    """
    Render an exact cited page to PNG bytes so users can visually verify the citation
    even if the browser PDF viewer thumbnail highlight is off due to continuous scroll.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None

    p = Path(file_path)
    if not p.exists():
        return None
    if page_1based < 1:
        return None

    try:
        doc = fitz.open(p)
        try:
            idx = page_1based - 1
            if idx < 0 or idx >= doc.page_count:
                return None
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_graph(persist_dir: str, embedding_model: str, cache_version: int = 2):
    """
    Build and cache the RAG graph.
    
    Args:
        persist_dir: Directory containing the vectorstore
        embedding_model: Embedding model name
        cache_version: Increment this to force cache invalidation when code changes
    
    Returns:
        RAG graph or None if vectorstore not found
    """
    rt = RetrieverTool(persist_directory=persist_dir, embedding_model=embedding_model)
    if not rt.load_vectorstore():
        return None  # Return None instead of raising - let UI handle it

    retrieve_tool = rt.get_retriever_tool()

    # Well picks: best-effort file discovery (users can still query via vectorstore too,
    # but deterministic tool is better for exhaustive lists).
    candidates = [
        Path("spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("../spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("Well_picks_Volve_v1.dat"),
    ]
    dat_path = next((str(p) for p in candidates if p.exists()), str(Path.cwd() / "Well_picks_Volve_v1.dat"))

    tools = [
        WellPicksTool(dat_path=dat_path).get_tool(),
        retrieve_tool,
    ]

    # Optional caches (if present they’ll be used for deterministic lookup + citations)
    section_index = Path(persist_dir) / "section_index.json"
    if section_index.exists():
        tools.append(SectionLookupTool(str(section_index)).get_tool())

    petro_cache = Path(persist_dir) / "petro_params_cache.json"
    if petro_cache.exists():
        tools.append(PetroParamsTool(str(petro_cache)).get_tool())

        # One-shot formations + properties tool
        try:
            tools.append(
                FormationPropertiesTool(
                    well_picks_dat_path=dat_path,
                    petro_params_cache_path=str(petro_cache),
                ).get_tool()
            )
        except Exception:
            pass

    # Evaluation parameters tool (uses its own cache)
    eval_cache = Path(persist_dir) / "eval_params_cache.json"
    if eval_cache.exists():
        try:
            tools.append(EvalParamsTool(str(eval_cache)).get_tool())
        except Exception:
            pass

    # Structured facts tool (notes/narrative numeric statements)
    facts_cache = Path(persist_dir) / "facts_cache.json"
    if facts_cache.exists():
        try:
            tools.append(StructuredFactsTool(str(facts_cache)).get_tool())
        except Exception:
            pass

    return build_rag_graph(tools)


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
        persist_dir = st.text_input("Vectorstore dir", value="./data/vectorstore")
        embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
        st.caption("Index build remains CLI-based: `python -m src.main --build-index`.")
        
        # Debug: show current working directory and vectorstore path
        cwd = Path.cwd()
        vs_path = Path(persist_dir)
        vs_abs = vs_path.resolve()
        st.caption(f"🔍 CWD: {cwd}")
        st.caption(f"🔍 Vectorstore: {vs_abs}")
        st.caption(f"🔍 Exists: {vs_abs.exists()}")

        # Also emit to the process logs for Streamlit Cloud visibility
        logger = logging.getLogger(__name__)
        logger.info(f"CWD: {cwd}")
        logger.info(f"Vectorstore path: {vs_abs}")
        logger.info(f"Vectorstore exists: {vs_abs.exists()}")

    # Chat history (multi-turn)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "viewer" not in st.session_state:
        st.session_state.viewer = {"path": None, "page": None}

    col_left, col_right = st.columns([0.55, 0.45], gap="large")

    with col_left:
        st.subheader("Chat")
        st.caption("Context is preserved in this chat: you can answer clarifications (e.g., just “matrix density”) and the prior well/formation will be remembered.")

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
                            st.caption(
                                f"`{c.source_path}`"
                                + (f" (pages {c.page_start}-{c.page_end})" if c.page_start else "")
                            )
                        with cols[1]:
                            label = f"View page {page}" if page else "View"
                            # Add idx component to avoid duplicate Streamlit keys when the same
                            # citation repeats across reruns or messages.
                            key_suffix = f"{len(st.session_state.messages)}_{msg_idx}_{0}_{c.source_path}_{page}"
                            if st.button(label, key=f"view_{key_suffix}"):
                                st.session_state.viewer = {"path": c.source_path, "page": page}

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
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                result = graph.invoke({"messages": st.session_state.messages})
                answer = result["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

    with col_right:
        st.subheader("PDF Viewer")
        vp = st.session_state.viewer.get("path")
        vpage = st.session_state.viewer.get("page")
        if vp:
            # Exact citation preview (always correct page)
            if isinstance(vpage, int) and vpage > 0:
                png = _render_pdf_page_png(vp, vpage)
                if png:
                    st.caption(f"Cited page preview (page {vpage})")
                    st.image(png, use_container_width=True)
            st.markdown(_pdf_iframe(vp, vpage), unsafe_allow_html=True)
        else:
            st.info("Click **View page** next to a source to open it here.")


if __name__ == "__main__":
    main()


