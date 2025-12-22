"""
Citation parsing logic for extracting source references from answers.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Citation:
    """Represents a citation with source path and page range."""
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

