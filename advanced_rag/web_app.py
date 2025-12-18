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

# Configure logging to ensure messages are visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing config
)

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


def _clean_source_path(source_path: str) -> str:
    """
    Clean up source path to show a user-friendly filename or relative path.
    Removes local Windows paths and shows just the filename or well/filename.
    
    Args:
        source_path: Original source path (may be full Windows path)
        
    Returns:
        Cleaned path (e.g., "15_9-F-5/PETROPHYSICAL_REPORT_1.PDF" or just filename)
    """
    if not source_path:
        return source_path
    
    # Handle Windows paths and relative paths
    path_str = source_path.replace('\\', '/')
    parts = [p for p in path_str.split('/') if p and p != '..' and p != '.']
    
    # Remove common prefixes like "C:", "Users", "Downloads", "spwla_volve-main"
    filtered_parts = []
    skip_next = False
    for i, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        # Skip drive letters, common user dirs
        if part.endswith(':') or part.lower() in ['users', 'downloads', 'spwla_volve-main']:
            continue
        # Keep well directories and filenames
        if re.match(r'^\d+[\s_/-]*\d+', part) or part.lower().endswith('.pdf'):
            filtered_parts.append(part)
    
    # If we have well directory and filename, return "well_dir/filename"
    if len(filtered_parts) >= 2:
        return '/'.join(filtered_parts[-2:])
    # Otherwise just return the filename
    elif filtered_parts:
        return filtered_parts[-1]
    else:
        # Fallback: just get the filename from the original path
        return Path(source_path).name


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
                    # Extract all files, handling duplicates by keeping the last one
                    extracted_files = set()
                    for member in zip_ref.namelist():
                        if member.endswith('/') or not member.lower().endswith('.pdf'):
                            continue
                        # Get just the filename (handle any path components)
                        filename = Path(member).name
                        dest_path = target_dir / filename
                        # Extract to destination
                        with zip_ref.open(member) as source:
                            with open(dest_path, 'wb') as target:
                                target.write(source.read())
                        extracted_files.add(filename)
                        logger.debug(f"Extracted: {filename}")
            
            # Verify extraction was successful by checking for PDF files
            pdf_files = list(target_dir.glob("*.pdf"))
            pdf_count = len(pdf_files)
            if pdf_count == 0:
                logger.error(f"No PDFs found after extraction in {target_dir}")
                # List what's actually in the directory
                all_files = list(target_dir.iterdir())
                logger.error(f"Directory contents: {[f.name for f in all_files[:10]]}")
                return False
            
            logger.info(f"Extraction complete. Found {pdf_count} PDF files in {target_dir}")
            logger.info(f"Sample PDFs: {[f.name for f in pdf_files[:5]]}")
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
    
    # Auto-fix: Update old repository name (VolveRAG) to new name (volverag)
    if pdfs_url and "VolveRAG" in pdfs_url:
        pdfs_url = pdfs_url.replace("VolveRAG", "volverag")
        logger.info(f"Auto-corrected repository name in PDFS_URL: {pdfs_url}")
    
    if not pdfs_url:
        return False  # No URL configured, can't download
    
    # Download and extract
    return _download_and_extract_pdfs(pdfs_url, pdfs_dir)


def _find_pdf_file(file_path: str, pdfs_dir: Path) -> Optional[Path]:
    """
    Find a PDF file by name, searching in the downloaded PDFs directory.
    Handles both original filenames and unique filenames with well prefix.
    
    Args:
        file_path: Original file path (may be a local Windows path or relative path)
                  e.g., "..\\spwla_volve-main\\15_9-F-5\\PETROPHYSICAL_REPORT_1.PDF"
        pdfs_dir: Directory where PDFs are stored
        
    Returns:
        Path to the PDF file if found, None otherwise
    """
    original_path = Path(file_path)
    
    # Clean up path - handle backslashes and relative paths
    path_str = str(original_path).replace('\\', '/')
    parts = [p for p in path_str.split('/') if p and p != '..' and p != '.']
    
    # Extract well directory and filename from path like:
    # "spwla_volve-main/15_9-F-5/PETROPHYSICAL_REPORT_1.PDF" or
    # "15_9-F15D/PETROPHYSICAL_REPORT_1.PDF"
    filename = parts[-1] if parts else original_path.name
    well_dir = None
    if len(parts) >= 2:
        # Try to find well directory (usually contains numbers and dashes)
        for part in parts[-2::-1]:  # Check parts from end backwards
            if re.match(r'^\d+[\s_/-]*\d+', part):  # Matches patterns like "15_9-F-5", "15_9-F15D"
                well_dir = part
                break
    
    logger.debug(f"[PDF_FIND] Looking for PDF: filename='{filename}', well_dir='{well_dir}', original_path='{file_path}', pdfs_dir='{pdfs_dir}'")
    
    # If the original path exists (local development), use it
    if original_path.exists():
        logger.debug(f"[PDF_FIND] Found PDF at original path: {original_path}")
        return original_path
    
    # Search in PDFs directory
    if pdfs_dir.exists():
        # Strategy 1: Try unique filename format (well_dir_filename.pdf)
        if well_dir:
            unique_name = f"{well_dir}_{filename}"
            unique_match = pdfs_dir / unique_name
            if unique_match.exists():
                logger.debug(f"[PDF_FIND] Found PDF via unique name: {unique_match}")
                return unique_match
        
        # Strategy 2: Try exact filename match
        exact_match = pdfs_dir / filename
        if exact_match.exists():
            logger.debug(f"[PDF_FIND] Found PDF at exact match: {exact_match}")
            return exact_match
        
        # Strategy 3: Try recursive search (handles subdirectories)
        matches = list(pdfs_dir.glob(f"**/{filename}"))
        if matches:
            logger.debug(f"[PDF_FIND] Found PDF via recursive search: {matches[0]}")
            return matches[0]
        
        # Strategy 4: Try case-insensitive search with unique name
        if well_dir:
            unique_pattern = f"{well_dir}_*{filename}"
            unique_matches = list(pdfs_dir.glob(unique_pattern))
            if unique_matches:
                logger.debug(f"[PDF_FIND] Found PDF via unique pattern: {unique_matches[0]}")
                return unique_matches[0]
        
        # Strategy 5: Try case-insensitive search by filename only
        for pdf_file in pdfs_dir.glob("*.pdf"):
            if pdf_file.name.lower() == filename.lower():
                logger.debug(f"[PDF_FIND] Found PDF via case-insensitive search: {pdf_file}")
                return pdf_file
        
        # Strategy 6: Try partial match (filename contains the target filename)
        filename_lower = filename.lower()
        for pdf_file in pdfs_dir.glob("*.pdf"):
            if filename_lower in pdf_file.name.lower() or pdf_file.name.lower() in filename_lower:
                logger.debug(f"[PDF_FIND] Found PDF via partial match: {pdf_file}")
                return pdf_file
        
        # Log what PDFs are actually available for debugging
        available_pdfs = list(pdfs_dir.glob("*.pdf"))[:10]  # First 10 for logging
        logger.debug(f"[PDF_FIND] PDF not found. Available PDFs (sample): {[p.name for p in available_pdfs]}")
    else:
        logger.debug(f"[PDF_FIND] PDFs directory does not exist: {pdfs_dir}")
    
    return None


def _pdf_full_viewer(file_path: str, initial_page: Optional[int] = None) -> str:
    """
    Display full PDF with page navigation using PDF.js.
    Users can scroll through all pages without downloading.
    Includes optional download button.
    """
    pdfs_dir = Path(__file__).resolve().parent / "data" / "pdfs"
    logger.info(f"[PDF_VIEWER] Looking for PDF: original_path='{file_path}', pdfs_dir='{pdfs_dir}'")
    
    p = _find_pdf_file(file_path, pdfs_dir)
    
    if p is None or not p.exists():
        filename = Path(file_path).name
        clean_path = _clean_source_path(file_path)
        return f"""
        <div style="padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <p style="font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">📄 PDF not available</p>
            <p style="margin-bottom: 5px;">File: <code style="background-color: #e8e8e8; padding: 2px 6px; border-radius: 3px;">{clean_path}</code></p>
        </div>
        """

    logger.info(f"[PDF_VIEWER] Found PDF: {p}, initial_page={initial_page}")
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        import hashlib
        unique_id = hashlib.md5(str(p).encode()).hexdigest()[:8]
        b64_escaped = b64.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        filename = p.name
        
        start_page = initial_page if initial_page else 1
        
        return f"""
        <div id="pdf-container-{unique_id}" style="border: 1px solid #ddd; border-radius: 5px; background-color: #f5f5f5;">
            <div id="pdf-controls-{unique_id}" style="padding: 10px; background-color: #fff; border-bottom: 1px solid #ddd; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <button id="prev-btn-{unique_id}" onclick="changePage(-1)" style="padding: 0.4rem 0.8rem; background-color: #1f77b4; color: white; border: none; border-radius: 0.25rem; cursor: pointer;">◀ Previous</button>
                <span id="page-info-{unique_id}" style="font-weight: 500;">Page <span id="current-page-{unique_id}">1</span> of <span id="total-pages-{unique_id}">-</span></span>
                <button id="next-btn-{unique_id}" onclick="changePage(1)" style="padding: 0.4rem 0.8rem; background-color: #1f77b4; color: white; border: none; border-radius: 0.25rem; cursor: pointer;">Next ▶</button>
                <input type="number" id="page-input-{unique_id}" min="1" value="{start_page}" style="width: 60px; padding: 0.3rem; border: 1px solid #ddd; border-radius: 0.25rem;" onchange="goToPage(this.value)">
                <button onclick="goToPage(document.getElementById('page-input-{unique_id}').value)" style="padding: 0.4rem 0.8rem; background-color: #6c757d; color: white; border: none; border-radius: 0.25rem; cursor: pointer;">Go</button>
                <div style="margin-left: auto;">
                    <button onclick="downloadPDF()" style="padding: 0.4rem 0.8rem; background-color: #28a745; color: white; border: none; border-radius: 0.25rem; cursor: pointer;">⬇️ Download PDF</button>
                </div>
            </div>
            <div id="pdf-viewer-{unique_id}" style="padding: 10px; overflow-y: auto; max-height: 800px; text-align: center;"></div>
            <div id="pdf-loading-{unique_id}" style="text-align: center; padding: 20px;">Loading PDF...</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
        <script>
        (function() {{
            let pdfDoc = null;
            let currentPage = {start_page};
            const base64 = "{b64_escaped}";
            const filename = "{filename}";
            const containerId = "pdf-container-{unique_id}";
            const viewerId = "pdf-viewer-{unique_id}";
            const loadingId = "pdf-loading-{unique_id}";
            const pageInfoId = "current-page-{unique_id}";
            const totalPagesId = "total-pages-{unique_id}";
            const pageInputId = "page-input-{unique_id}";
            const prevBtnId = "prev-btn-{unique_id}";
            const nextBtnId = "next-btn-{unique_id}";
            
            function renderPage(pageNum) {{
                if (!pdfDoc) return;
                
                pdfDoc.getPage(pageNum).then(function(page) {{
                    const scale = 1.5;
                    const viewport = page.getViewport({{scale: scale}});
                    
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;
                    canvas.style.marginBottom = '10px';
                    canvas.style.border = '1px solid #ddd';
                    canvas.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                    
                    const viewer = document.getElementById(viewerId);
                    viewer.innerHTML = '';
                    viewer.appendChild(canvas);
                    
                    const renderContext = {{
                        canvasContext: context,
                        viewport: viewport
                    }};
                    
                    page.render(renderContext).promise.then(function() {{
                        currentPage = pageNum;
                        document.getElementById(pageInfoId).textContent = pageNum;
                        document.getElementById(pageInputId).value = pageNum;
                        document.getElementById(prevBtnId).disabled = (pageNum <= 1);
                        document.getElementById(nextBtnId).disabled = (pageNum >= pdfDoc.numPages);
                    }});
                }});
            }}
            
            window.changePage = function(delta) {{
                const newPage = currentPage + delta;
                if (newPage >= 1 && newPage <= pdfDoc.numPages) {{
                    renderPage(newPage);
                }}
            }};
            
            window.goToPage = function(pageNum) {{
                const num = parseInt(pageNum);
                if (num >= 1 && num <= pdfDoc.numPages) {{
                    renderPage(num);
                }}
            }};
            
            window.downloadPDF = function() {{
                try {{
                    const binaryString = atob(base64);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}
                    const blob = new Blob([bytes], {{ type: 'application/pdf' }});
                    const url = URL.createObjectURL(blob);
                    const downloadLink = document.createElement('a');
                    downloadLink.href = url;
                    downloadLink.download = filename;
                    downloadLink.click();
                    URL.revokeObjectURL(url);
                }} catch (error) {{
                    console.error('Failed to download PDF:', error);
                    // Fallback to data URI
                    const downloadLink = document.createElement('a');
                    downloadLink.href = 'data:application/pdf;base64,' + base64;
                    downloadLink.download = filename;
                    downloadLink.click();
                }}
            }};
            
            function loadPDF() {{
                const viewer = document.getElementById(viewerId);
                const loading = document.getElementById(loadingId);
                
                if (!viewer) return;
                
                try {{
                    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                    
                    const binaryString = atob(base64);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}
                    
                    pdfjsLib.getDocument({{data: bytes}}).promise.then(function(pdf) {{
                        pdfDoc = pdf;
                        loading.style.display = 'none';
                        document.getElementById(totalPagesId).textContent = pdf.numPages;
                        document.getElementById(pageInputId).max = pdf.numPages;
                        renderPage({start_page});
                    }}).catch(function(error) {{
                        console.error('Error loading PDF:', error);
                        loading.style.display = 'none';
                        viewer.innerHTML = '<div style="padding: 20px; text-align: center;"><p>Failed to load PDF.</p></div>';
                    }});
                }} catch (error) {{
                    console.error('Failed to load PDF:', error);
                    loading.style.display = 'none';
                    viewer.innerHTML = '<div style="padding: 20px; text-align: center;"><p>Error loading PDF viewer.</p></div>';
                }}
            }}
            
            if (typeof pdfjsLib !== 'undefined') {{
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', loadPDF);
                }} else {{
                    setTimeout(loadPDF, 100);
                }}
            }} else {{
                window.addEventListener('load', function() {{
                    setTimeout(loadPDF, 500);
                }});
            }}
        }})();
        </script>
        """
    except Exception as e:
        logger.error(f"Failed to load PDF {p}: {e}")
        return f'<div style="padding: 20px; text-align: center;">Error loading PDF: {e}</div>'


def _get_pdf_data_uri(file_path: str) -> Optional[str]:
    """
    Get a data URI for the PDF file that can be opened in a new tab.
    
    Args:
        file_path: Original file path
        
    Returns:
        Data URI string if PDF found, None otherwise
    """
    pdfs_dir = Path(__file__).resolve().parent / "data" / "pdfs"
    p = _find_pdf_file(file_path, pdfs_dir)
    
    if p is None or not p.exists():
        return None
    
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:application/pdf;base64,{b64}"
    except Exception as e:
        logger.error(f"Failed to create PDF data URI for {p}: {e}")
        return None


def _pdf_iframe(file_path: str, page: Optional[int]) -> str:
    """
    Display PDF in object tag, handling both local and downloaded paths.
    Chrome blocks data URIs in iframes, so we use object tag instead.
    """
    # Try to find the PDF file
    pdfs_dir = Path(__file__).resolve().parent / "data" / "pdfs"
    logger.info(f"[PDF_VIEWER] Looking for PDF: original_path='{file_path}', pdfs_dir='{pdfs_dir}'")
    
    p = _find_pdf_file(file_path, pdfs_dir)
    
    if p is None or not p.exists():
        # PDF not available - show helpful message
        filename = Path(file_path).name
        clean_path = _clean_source_path(file_path)
        logger.warning(f"[PDF_VIEWER] PDF not found: original='{file_path}', cleaned='{clean_path}', pdfs_dir='{pdfs_dir}'")
        
        # Check if PDFs directory exists and list what's available
        available_info = ""
        if pdfs_dir.exists():
            available_pdfs = list(pdfs_dir.glob("*.pdf"))[:5]
            if available_pdfs:
                available_info = f"<br>Available PDFs (sample): {', '.join([p.name for p in available_pdfs])}"
        
        return f"""
        <div style="padding: 20px; text-align: center; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <p style="font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">📄 PDF not available</p>
            <p style="margin-bottom: 5px;">File: <code style="background-color: #e8e8e8; padding: 2px 6px; border-radius: 3px;">{clean_path}</code></p>
            <p style="color: #666; font-size: 0.9em; margin-top: 10px;">
                The PDF file could not be located.<br>
                If running on Streamlit Cloud, ensure <code>PDFS_URL</code> is set in secrets.<br>
                For local use, ensure the Volve dataset is accessible.{available_info}
            </p>
        </div>
        """

    # PDF found - display it using PDF.js (bypasses Chrome blocking)
    logger.info(f"[PDF_VIEWER] Found PDF: {p}, page={page}")
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        
        # Create a unique ID based on file path hash
        import hashlib
        unique_id = hashlib.md5(str(p).encode()).hexdigest()[:8]
        
        # Escape the base64 string for use in JavaScript
        b64_escaped = b64.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        
        # Use PDF.js from CDN to render PDF (bypasses Chrome blocking)
        target_page = page if page else 1
        
        return f"""
        <div id="pdf-container-{unique_id}" style="border: 1px solid #ddd; border-radius: 5px; overflow: auto; height: 900px; background-color: #f5f5f5;">
            <div id="pdf-viewer-{unique_id}" style="padding: 10px;"></div>
            <div id="pdf-loading-{unique_id}" style="text-align: center; padding: 20px;">Loading PDF...</div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
        <script>
        (function() {{
            function loadPDF() {{
                const base64 = "{b64_escaped}";
                const containerId = "pdf-container-{unique_id}";
                const viewerId = "pdf-viewer-{unique_id}";
                const loadingId = "pdf-loading-{unique_id}";
                const targetPage = {target_page};
                
                const container = document.getElementById(containerId);
                const viewer = document.getElementById(viewerId);
                const loading = document.getElementById(loadingId);
                
                if (!viewer || !container) {{
                    console.error('PDF container not found');
                    return;
                }}
                
                try {{
                    // Set up PDF.js
                    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                    
                    // Convert base64 to Uint8Array
                    const binaryString = atob(base64);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}
                    
                    // Load PDF
                    pdfjsLib.getDocument({{data: bytes}}).promise.then(function(pdf) {{
                        loading.style.display = 'none';
                        
                        // Render the target page
                        const pageNum = Math.min(Math.max(1, targetPage), pdf.numPages);
                        pdf.getPage(pageNum).then(function(page) {{
                            const scale = 1.5;
                            const viewport = page.getViewport({{scale: scale}});
                            
                            const canvas = document.createElement('canvas');
                            const context = canvas.getContext('2d');
                            canvas.height = viewport.height;
                            canvas.width = viewport.width;
                            
                            viewer.innerHTML = '';
                            viewer.appendChild(canvas);
                            
                            const renderContext = {{
                                canvasContext: context,
                                viewport: viewport
                            }};
                            
                            page.render(renderContext).promise.then(function() {{
                                console.log('PDF page rendered successfully');
                            }});
                        }});
                    }}).catch(function(error) {{
                        console.error('Error loading PDF:', error);
                        loading.style.display = 'none';
                        viewer.innerHTML = '<div style="padding: 20px; text-align: center;"><p>Failed to load PDF.</p><p><a href="data:application/pdf;base64,' + base64 + '" download="{p.name}">Download PDF</a></p></div>';
                    }});
                }} catch (error) {{
                    console.error('Failed to load PDF:', error);
                    loading.style.display = 'none';
                    viewer.innerHTML = '<div style="padding: 20px; text-align: center;"><p>Error loading PDF viewer.</p><p><a href="data:application/pdf;base64,' + base64 + '" download="{p.name}">Download PDF</a></p></div>';
                }}
            }}
            
            // Wait for PDF.js to load, then load PDF
            if (typeof pdfjsLib !== 'undefined') {{
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', loadPDF);
                }} else {{
                    setTimeout(loadPDF, 100);
                }}
            }} else {{
                // Wait for PDF.js script to load
                window.addEventListener('load', function() {{
                    setTimeout(loadPDF, 500);
                }});
            }}
        }})();
        </script>
        """
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
                
                # Normalize paths (handle both Windows \ and Unix /)
                normalized_files = []
                root_dirs = set()
                for file_path in all_files:
                    # Skip empty entries
                    if not file_path or file_path.endswith('/') or file_path.endswith('\\'):
                        continue
                    # Normalize path separators
                    normalized = file_path.replace('\\', '/')
                    normalized_files.append((file_path, normalized))
                    # Get the first directory component
                    parts = normalized.split('/')
                    if len(parts) > 1:
                        root_dirs.add(parts[0])
                
                # If there's a single root directory, extract its contents
                if len(root_dirs) == 1:
                    root_dir = list(root_dirs)[0]
                    logger.info(f"ZIP contains root directory '{root_dir}'. Extracting contents...")
                    
                    # Manually extract files, stripping the root_dir prefix
                    extracted_count = 0
                    for orig_path, norm_path in normalized_files:
                        # Check if this file is under the root directory
                        if not norm_path.startswith(root_dir + '/'):
                            continue
                        
                        # Get relative path within root_dir
                        rel_path = norm_path[len(root_dir) + 1:]
                        if not rel_path:
                            continue
                        
                        # Create destination path
                        dest_path = target_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Extract the file
                        try:
                            with zip_ref.open(orig_path) as source:
                                with open(dest_path, 'wb') as dest:
                                    dest.write(source.read())
                            extracted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to extract {orig_path}: {e}")
                    
                    logger.info(f"Extracted {extracted_count} files from '{root_dir}' directory")
                else:
                    # Files are at root level OR have mixed paths - need to handle Windows-style paths
                    # Check if files have a common prefix like "vectorstore/"
                    common_prefix = None
                    for orig_path, norm_path in normalized_files:
                        parts = norm_path.split('/')
                        if len(parts) > 1:
                            if common_prefix is None:
                                common_prefix = parts[0]
                            elif common_prefix != parts[0]:
                                common_prefix = None
                                break
                    
                    if common_prefix:
                        # Files are in a subdirectory - manually extract to handle Windows paths
                        logger.info(f"ZIP contains files in '{common_prefix}' subdirectory. Extracting and moving contents...")
                        # Manually extract files to handle Windows-style paths correctly
                        for orig_path, norm_path in normalized_files:
                            # Get the relative path within the common_prefix
                            if norm_path.startswith(common_prefix + '/'):
                                rel_path = norm_path[len(common_prefix) + 1:]
                            else:
                                rel_path = norm_path
                            
                            # Skip if no relative path (shouldn't happen)
                            if not rel_path:
                                continue
                            
                            # Create destination path
                            dest_path = target_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Extract the file
                            with zip_ref.open(orig_path) as source:
                                with open(dest_path, 'wb') as dest:
                                    dest.write(source.read())
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
    
    # Auto-fix: Update old repository name (VolveRAG) to new name (volverag)
    if vectorstore_url and "VolveRAG" in vectorstore_url:
        vectorstore_url = vectorstore_url.replace("VolveRAG", "volverag")
        logger.info(f"Auto-corrected repository name in URL: {vectorstore_url}")
    
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
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking..."):
                result = graph.invoke({"messages": st.session_state.messages})
                answer = result["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Debug information expander
                with st.expander("🔍 Debug Info (click to view diagnostics)", expanded=False):
                    st.write("**Query:**", user_input)
                    
                    # Check if petro cache exists
                    vectorstore_dir = Path(__file__).resolve().parent / "data" / "vectorstore"
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
            # Exact citation preview (always correct page)
            if isinstance(vpage, int) and vpage > 0:
                png = _render_pdf_page_png(vp, vpage)
                if png:
                    st.caption(f"Cited page preview (page {vpage})")
                    st.image(png, width='stretch')
                    
                    # Add button to open full PDF in new tab using blob URL, with download fallback
                    pdf_data_uri = _get_pdf_data_uri(vp)
                    if pdf_data_uri:
                        # Use JavaScript blob URL to open PDF in new tab (Chrome allows this)
                        import streamlit.components.v1 as components
                        clean_path = _clean_source_path(vp)
                        filename = Path(vp).name
                        # Escape the base64 string for JavaScript
                        b64 = pdf_data_uri.split(',')[1]  # Extract base64 part
                        b64_escaped = b64.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
                        
                        components.html(
                            f"""
                            <div style="margin-top: 0.5rem;">
                                <button onclick="(function() {{
                                    const base64 = '{b64_escaped}';
                                    try {{
                                        const binaryString = atob(base64);
                                        const bytes = new Uint8Array(binaryString.length);
                                        for (let i = 0; i < binaryString.length; i++) {{
                                            bytes[i] = binaryString.charCodeAt(i);
                                        }}
                                        const blob = new Blob([bytes], {{ type: 'application/pdf' }});
                                        const url = URL.createObjectURL(blob);
                                        const newWindow = window.open(url, '_blank');
                                        if (newWindow) {{
                                            // Clean up blob URL after a delay
                                            setTimeout(() => URL.revokeObjectURL(url), 1000);
                                        }} else {{
                                            // Pop-up blocked, offer download instead
                                            const downloadLink = document.createElement('a');
                                            downloadLink.href = url;
                                            downloadLink.download = '{filename}';
                                            downloadLink.click();
                                            URL.revokeObjectURL(url);
                                        }}
                                    }} catch (error) {{
                                        console.error('Failed to open PDF:', error);
                                        // Fallback to download
                                        const downloadLink = document.createElement('a');
                                        downloadLink.href = 'data:application/pdf;base64,' + base64;
                                        downloadLink.download = '{filename}';
                                        downloadLink.click();
                                    }}
                                }})()" 
                                style="padding: 0.5rem 1rem; background-color: #1f77b4; color: white; border: none; border-radius: 0.25rem; font-weight: 500; cursor: pointer; font-size: 0.9rem; margin-right: 0.5rem;">
                                    📄 Open full PDF in new tab
                                </button>
                                <a href="{pdf_data_uri}" download="{filename}" 
                                   style="display: inline-block; padding: 0.5rem 1rem; background-color: #6c757d; color: white; text-decoration: none; border-radius: 0.25rem; font-weight: 500; font-size: 0.9rem;">
                                    ⬇️ Download PDF
                                </a>
                            </div>
                            """,
                            height=60
                        )
                    else:
                        st.caption("⚠️ Full PDF not available")
                else:
                    st.info("PDF page preview not available. The PDF file may not be accessible.")
            else:
                st.info("Click **View page** next to a source to open it here.")
        else:
            st.info("Click **View page** next to a source to open it here.")


if __name__ == "__main__":
    main()


