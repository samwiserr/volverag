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
import zipfile
import urllib.request
import tempfile
import shutil
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

logger = logging.getLogger(__name__)


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


def _download_and_extract_pdfs(zip_url: str, target_dir: Path) -> bool:
    """
    Download PDFs ZIP from URL and extract to target directory.
    
    Args:
        zip_url: URL to the ZIP file containing PDFs
        target_dir: Directory where PDFs should be extracted
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading PDFs from {zip_url}...")
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_zip_path = tmp_file.name
        
        try:
            # Download with progress
            urllib.request.urlretrieve(zip_url, tmp_zip_path)
            logger.info(f"Download complete. Extracting PDFs to {target_dir}...")
            
            # Extract ZIP
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                # Get list of all file paths in ZIP
                all_files = zip_ref.namelist()
                
                # Check if all files are under a single root directory
                root_dirs = set()
                for file_path in all_files:
                    # Skip empty entries
                    if not file_path or file_path.endswith('/'):
                        continue
                    # Get the first directory component
                    parts = file_path.split('/')
                    if len(parts) > 1:
                        root_dirs.add(parts[0])
                
                # If there's a single root directory, extract its contents
                if len(root_dirs) == 1:
                    root_dir = list(root_dirs)[0]
                    logger.info(f"ZIP contains root directory '{root_dir}'. Extracting contents...")
                    
                    # Extract to temporary location first
                    with tempfile.TemporaryDirectory() as tmp_extract:
                        zip_ref.extractall(tmp_extract)
                        source_dir = Path(tmp_extract) / root_dir
                        
                        # Move contents from root_dir to target_dir
                        if source_dir.exists():
                            for item in source_dir.iterdir():
                                dest = target_dir / item.name
                                if item.is_dir():
                                    if dest.exists():
                                        shutil.rmtree(dest)
                                    shutil.copytree(item, dest)
                                else:
                                    shutil.copy2(item, dest)
                else:
                    # Files are at root level, extract directly
                    logger.info("ZIP contains files at root level. Extracting directly...")
                    zip_ref.extractall(target_dir)
            
            # Verify extraction was successful by checking for PDF files
            pdf_count = len(list(target_dir.glob("**/*.pdf")))
            if pdf_count == 0:
                logger.error(f"No PDFs found after extraction in {target_dir}")
                return False
            
            logger.info(f"Extraction complete. Found {pdf_count} PDF files in {target_dir}")
            return True
            
        finally:
            # Clean up temporary ZIP file
            if Path(tmp_zip_path).exists():
                Path(tmp_zip_path).unlink()
                
    except Exception as e:
        logger.error(f"Failed to download/extract PDFs: {e}", exc_info=True)
        return False


def _ensure_pdfs_available(pdfs_dir: Path) -> bool:
    """
    Ensure PDFs are available, downloading from GitHub Releases if needed.
    
    Args:
        pdfs_dir: Directory where PDFs should be stored
        
    Returns:
        True if PDFs are available, False otherwise
    """
    # Check if PDFs already exist
    if pdfs_dir.exists() and any(pdfs_dir.glob("**/*.pdf")):
        return True
    
    # Try to get PDFS_URL from environment or Streamlit secrets
    pdfs_url = os.getenv("PDFS_URL")
    if not pdfs_url:
        try:
            if "PDFS_URL" in st.secrets:
                pdfs_url = str(st.secrets["PDFS_URL"])
        except Exception:
            pass
    
    if not pdfs_url:
        return False  # No URL configured, can't download
    
    # Download and extract
    return _download_and_extract_pdfs(pdfs_url, pdfs_dir)


def _find_pdf_file(file_path: str, pdfs_dir: Path) -> Optional[Path]:
    """
    Find a PDF file by name, searching in the downloaded PDFs directory.
    
    Args:
        file_path: Original file path (may be a local Windows path)
        pdfs_dir: Directory where PDFs are stored
        
    Returns:
        Path to the PDF file if found, None otherwise
    """
    original_path = Path(file_path)
    filename = original_path.name
    
    # If the original path exists, use it
    if original_path.exists():
        return original_path
    
    # Search in PDFs directory
    if pdfs_dir.exists():
        # Try exact match first
        exact_match = pdfs_dir / filename
        if exact_match.exists():
            return exact_match
        
        # Try recursive search
        matches = list(pdfs_dir.glob(f"**/{filename}"))
        if matches:
            return matches[0]
        
        # Try case-insensitive search
        for pdf_file in pdfs_dir.glob("**/*.pdf"):
            if pdf_file.name.lower() == filename.lower():
                return pdf_file
    
    return None


def _pdf_iframe(file_path: str, page: Optional[int]) -> str:
    """
    Display PDF in iframe, handling both local and downloaded paths.
    """
    # Try to find the PDF file
    pdfs_dir = Path(__file__).resolve().parent / "data" / "pdfs"
    p = _find_pdf_file(file_path, pdfs_dir)
    
    if p is None or not p.exists():
        # PDF not available - show helpful message
        filename = Path(file_path).name
        return f"""
        <div style="padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <p style="font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">📄 PDF not available</p>
            <p style="margin-bottom: 5px;">File: <code style="background-color: #e8e8e8; padding: 2px 6px; border-radius: 3px;">{filename}</code></p>
            <p style="color: #666; font-size: 0.9em; margin-top: 10px;">
                The PDF file could not be located.<br>
                If running on Streamlit Cloud, ensure <code>PDFS_URL</code> is set in secrets.<br>
                For local use, ensure the Volve dataset is accessible.
            </p>
        </div>
        """

    # PDF found - display it
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        # Best-effort: encourage the viewer to land at the top of the cited page
        # (reduces cases where continuous-scroll highlights the next thumbnail).
        frag = f"#page={page}&view=FitH&zoom=page-width" if page else ""
        return (
            f'<iframe src="data:application/pdf;base64,{b64}{frag}" '
            f'width="100%" height="900" type="application/pdf"></iframe>'
        )
    except Exception as e:
        logger.error(f"Failed to load PDF {p}: {e}")
        return f'<div style="padding: 20px; text-align: center;">Error loading PDF: {e}</div>'


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

    # Try to find the PDF file
    pdfs_dir = Path(__file__).resolve().parent / "data" / "pdfs"
    p = _find_pdf_file(file_path, pdfs_dir)
    
    if p is None or not p.exists():
        return None  # PDF not available (e.g., on Streamlit Cloud without PDFS_URL)
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


def _download_and_extract_vectorstore(zip_url: str, target_dir: Path) -> bool:
    """
    Download ZIP from URL and extract to target directory.
    Handles ZIP files with a single root directory or files at root.
    
    Args:
        zip_url: URL to the ZIP file
        target_dir: Directory where vectorstore should be extracted
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading vectorstore from {zip_url}...")
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Download to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_zip_path = tmp_file.name
        
        try:
            # Download with progress
            urllib.request.urlretrieve(zip_url, tmp_zip_path)
            logger.info(f"Download complete. Extracting to {target_dir}...")
            
            # Extract ZIP
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                # Get list of all file paths in ZIP
                all_files = zip_ref.namelist()
                
                # Check if all files are under a single root directory
                root_dirs = set()
                for file_path in all_files:
                    # Skip empty entries
                    if not file_path or file_path.endswith('/'):
                        continue
                    # Get the first directory component
                    parts = file_path.split('/')
                    if len(parts) > 1:
                        root_dirs.add(parts[0])
                
                # If there's a single root directory, extract its contents
                if len(root_dirs) == 1:
                    root_dir = list(root_dirs)[0]
                    logger.info(f"ZIP contains root directory '{root_dir}'. Extracting contents...")
                    
                    # Extract to temporary location first
                    with tempfile.TemporaryDirectory() as tmp_extract:
                        zip_ref.extractall(tmp_extract)
                        source_dir = Path(tmp_extract) / root_dir
                        
                        # Move contents from root_dir to target_dir
                        if source_dir.exists():
                            for item in source_dir.iterdir():
                                dest = target_dir / item.name
                                if item.is_dir():
                                    if dest.exists():
                                        shutil.rmtree(dest)
                                    shutil.copytree(item, dest)
                                else:
                                    shutil.copy2(item, dest)
                else:
                    # Files are at root level, extract directly
                    logger.info("ZIP contains files at root level. Extracting directly...")
                    zip_ref.extractall(target_dir)
            
            # Verify extraction was successful by checking for chroma.sqlite3
            chroma_db = target_dir / "chroma.sqlite3"
            if not chroma_db.exists():
                logger.error(f"Extraction incomplete: chroma.sqlite3 not found in {target_dir}")
                # List what was actually extracted for debugging
                extracted = list(target_dir.iterdir())
                logger.error(f"Extracted items: {[str(p.name) for p in extracted]}")
                return False
            
            logger.info(f"Extraction complete. Vectorstore ready at {target_dir}")
            logger.info(f"Found files: chroma.sqlite3, {len(list(target_dir.glob('*.json')))} cache files")
            return True
            
        finally:
            # Clean up temporary ZIP file
            if Path(tmp_zip_path).exists():
                Path(tmp_zip_path).unlink()
                
    except Exception as e:
        logger.error(f"Failed to download/extract vectorstore: {e}", exc_info=True)
        return False


def _ensure_vectorstore_available(persist_dir: str) -> bool:
    """
    Ensure vectorstore exists, downloading from URL if needed.
    
    Args:
        persist_dir: Directory where vectorstore should be located
        
    Returns:
        True if vectorstore is available, False otherwise
    """
    persist_path = Path(persist_dir)
    
    # Check if vectorstore already exists
    if persist_path.exists():
        # Check for key files that indicate a valid vectorstore
        chroma_db = persist_path / "chroma.sqlite3"
        if chroma_db.exists():
            return True
    
    # Try to get VECTORSTORE_URL from environment or Streamlit secrets
    vectorstore_url = os.getenv("VECTORSTORE_URL")
    if not vectorstore_url:
        try:
            if "VECTORSTORE_URL" in st.secrets:
                vectorstore_url = str(st.secrets["VECTORSTORE_URL"])
        except Exception:
            pass
    
    if not vectorstore_url:
        return False  # No URL configured, can't download
    
    # Download and extract
    return _download_and_extract_vectorstore(vectorstore_url, persist_path)


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

    # Start with retrieval tool; add WellPicksTool only if data/cache is available
    tools = [retrieve_tool]
    try:
        cache_path = Path(persist_dir) / "well_picks_cache.json"
        well_picks_tool = WellPicksTool(dat_path=dat_path, cache_path=str(cache_path))
        if getattr(well_picks_tool, "_rows", None):
            tools.insert(0, well_picks_tool.get_tool())
    except Exception as e:
        logger = __import__('logging').getLogger(__name__)
        logger.warning(f"[WEB_APP] WellPicksTool not available: {e}. Continuing without it.")

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
        # Default persist dir: resolve relative to this module so it works
        # both when running from `advanced_rag/` and when Streamlit runs the
        # module from the repo root (Streamlit Cloud clones to /mount/src/...).
        default_persist = str(Path(__file__).resolve().parent / "data" / "vectorstore")
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
            
            if vectorstore_url:
                with st.spinner("Downloading vectorstore... This may take a few minutes on first run."):
                    if _ensure_vectorstore_available(persist_dir):
                        st.success("Vectorstore downloaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to download vectorstore. Please check the VECTORSTORE_URL and try again.")
                        st.info(f"Attempted URL: {vectorstore_url}")
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
        pdfs_dir = Path(__file__).resolve().parent / "data" / "pdfs"
        if not _ensure_pdfs_available(pdfs_dir):
            # Show info in sidebar but don't block - app can work without PDFs
            pdfs_url = os.getenv("PDFS_URL")
            if not pdfs_url:
                try:
                    if "PDFS_URL" in st.secrets:
                        pdfs_url = str(st.secrets["PDFS_URL"])
                except Exception:
                    pass
            
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


