"""
Asset downloader for PDFs and vectorstore from GitHub Releases.
"""
import os
import zipfile
import urllib.request
import tempfile
import shutil
import logging
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)


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
        persist_dir: Directory where vectorstore should be stored
        
    Returns:
        True if vectorstore is available, False otherwise
    """
    # Check if vectorstore already exists
    persist_path = Path(persist_dir)
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
        logger.info(f"Auto-corrected repository name in VECTORSTORE_URL: {vectorstore_url}")
    
    if not vectorstore_url:
        return False  # No URL configured, can't download
    
    # Download and extract
    return _download_and_extract_vectorstore(vectorstore_url, persist_path)

