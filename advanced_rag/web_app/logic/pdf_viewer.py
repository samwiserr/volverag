"""
PDF viewer logic for displaying and navigating PDF documents.
"""
import base64
import re
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from .citation_parser import _clean_source_path

logger = logging.getLogger(__name__)


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
    # Resolve pdfs_dir relative to advanced_rag root (web_app/logic/pdf_viewer.py -> advanced_rag/)
    pdfs_dir = Path(__file__).resolve().parents[2] / "data" / "pdfs"
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
            <div id="pdf-viewer-{unique_id}" style="padding: 10px; overflow-y: auto; max-height: 800px; text-align: center; width: 100%;"></div>
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
                    // Calculate scale to fit container width (with some padding)
                    const viewer = document.getElementById(viewerId);
                    const containerWidth = viewer.clientWidth || 800; // Fallback to 800px
                    const padding = 40; // Padding on each side
                    const availableWidth = containerWidth - padding;
                    
                    // Get page dimensions at scale 1.0
                    const viewport1 = page.getViewport({{scale: 1.0}});
                    // Calculate scale to fit width
                    const scale = Math.min(availableWidth / viewport1.width, 2.0); // Max scale of 2.0
                    
                    const viewport = page.getViewport({{scale: scale}});
                    
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;
                    canvas.style.marginBottom = '10px';
                    canvas.style.border = '1px solid #ddd';
                    canvas.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                    canvas.style.maxWidth = '100%';
                    canvas.style.height = 'auto';
                    
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
    # Resolve pdfs_dir relative to advanced_rag root (web_app/logic/pdf_viewer.py -> advanced_rag/)
    pdfs_dir = Path(__file__).resolve().parents[2] / "data" / "pdfs"
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
    # Resolve pdfs_dir relative to advanced_rag root (web_app/logic/pdf_viewer.py -> advanced_rag/)
    pdfs_dir = Path(__file__).resolve().parents[2] / "data" / "pdfs"
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
    # Resolve pdfs_dir relative to advanced_rag root (web_app/logic/pdf_viewer.py -> advanced_rag/)
    pdfs_dir = Path(__file__).resolve().parents[2] / "data" / "pdfs"
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

