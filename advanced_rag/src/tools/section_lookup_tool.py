"""
Deterministic section lookup tool.

Problem it solves:
- RAG often retrieves Table-of-Contents (TOC) lines like:
  "Summary 15/9-F-11 B .......... 6"
  instead of the real "2.1 Summary 15/9-F-11 / 15/9-F-11 T2" section.

Approach:
- During indexing, build a persisted section index from per-page Documents:
  (heading -> extracted section text spanning until next heading).
- At query time, route section-like queries to this tool and bypass LLM summarization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Keep well separators, slash, dash
    s = re.sub(r"[^a-z0-9/\-\. ]+", "", s)
    return s


def _norm_compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _extract_query_well(query: str) -> Optional[str]:
    m = re.search(r"(15[\s_/-]*9[\s_/-]*(?:f[\s_/-]*)?\d+[a-z]?(?:[\s_/-]*t2)?)", query, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _is_toc_line(line: str) -> bool:
    # Dotted leader + page number at end
    if re.search(r"\.{4,}\s*\d+\s*$", line):
        return True
    # Very short heading + trailing number
    if re.search(r"\s{2,}\d+\s*$", line) and len(line.strip()) < 80:
        return True
    return False


@dataclass
class SectionEntry:
    heading: str
    heading_norm: str
    source: str
    level: int
    start_page: Optional[int]
    end_page: Optional[int]
    text: str


class SectionLookupTool:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Section index not found: {self.index_path}")
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self._entries: List[SectionEntry] = [SectionEntry(**e) for e in payload.get("sections", [])]
        logger.info(f"[OK] Loaded section index with {len(self._entries)} sections")

    @staticmethod
    def build_index(documents: List[Document], out_path: str) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Group per-page docs by source
        groups: Dict[str, List[Document]] = {}
        for d in documents:
            src = str(d.metadata.get("source", ""))
            if not src:
                continue
            # Skip well picks .dat
            if src.lower().endswith(".dat"):
                continue
            groups.setdefault(src, []).append(d)

        sections: List[SectionEntry] = []

        heading_re_num = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.{4,200})\s*$")
        heading_re_word = re.compile(r"^\s*(summary|introduction|conclusion|conclusions|results|discussion|abstract|executive summary)\b(.{0,200})$",
                                     re.IGNORECASE)

        for src, docs in groups.items():
            # Sort by page if present
            def page_key(doc: Document) -> int:
                p = doc.metadata.get("page", doc.metadata.get("page_number"))
                try:
                    return int(p)
                except Exception:
                    return 0

            docs_sorted = sorted(docs, key=page_key)

            # Build full text with page markers to help page range extraction
            parts: List[str] = []
            for d in docs_sorted:
                p = d.metadata.get("page", d.metadata.get("page_number"))
                try:
                    p_int = int(p)
                except Exception:
                    p_int = None
                if p_int is not None:
                    parts.append(f"\n[PAGE {p_int}]\n")
                parts.append(d.page_content or "")
                parts.append("\n")
            full_text = "".join(parts)

            # Walk lines with offsets, tracking current page
            headings: List[Tuple[int, int, str, Optional[int]]] = []  # (pos, level, heading, page)
            cur_page: Optional[int] = None
            pos = 0
            for line in full_text.splitlines(True):  # keepends
                m_page = re.match(r"^\[PAGE\s+(\d+)\]\s*$", line.strip())
                if m_page:
                    cur_page = int(m_page.group(1))
                    pos += len(line)
                    continue

                raw = line.rstrip("\n")
                if not raw.strip():
                    pos += len(line)
                    continue

                if _is_toc_line(raw):
                    pos += len(line)
                    continue

                m_num = heading_re_num.match(raw)
                if m_num:
                    num = m_num.group(1).strip()
                    title = m_num.group(2).strip()
                    level = num.count(".") + 1
                    heading = f"{num} {title}"
                    headings.append((pos, level, heading, cur_page))
                    pos += len(line)
                    continue

                m_word = heading_re_word.match(raw)
                if m_word:
                    heading = raw.strip()
                    headings.append((pos, 1, heading, cur_page))
                    pos += len(line)
                    continue

                pos += len(line)

            if not headings:
                continue

            # Determine end offsets by next heading with level <= current
            for i, (hpos, lvl, heading, spage) in enumerate(headings):
                end = len(full_text)
                epage: Optional[int] = None
                for j in range(i + 1, len(headings)):
                    npos, nlvl, _nh, npg = headings[j]
                    if nlvl <= lvl:
                        end = npos
                        epage = npg
                        break

                text = full_text[hpos:end].strip()
                if len(text) < 40:
                    continue

                # Determine page range from page markers inside section text
                pages = [int(p) for p in re.findall(r"\[PAGE\s+(\d+)\]", text)]
                start_page = min(pages) if pages else spage
                end_page = max(pages) if pages else (epage or spage)

                # Remove page markers from returned text
                text_clean = re.sub(r"\[PAGE\s+\d+\]\s*", "", text)
                # Cap to protect tool output size
                if len(text_clean) > 80000:
                    text_clean = text_clean[:80000] + "\n...[truncated]..."

                sections.append(
                    SectionEntry(
                        heading=heading,
                        heading_norm=_norm_compact(heading),
                        source=src,
                        level=lvl,
                        start_page=start_page,
                        end_page=end_page,
                        text=text_clean.strip(),
                    )
                )

        # Persist with md5 over sources list to detect rebuild needs
        md5 = hashlib.md5(("|".join(sorted(groups.keys()))).encode("utf-8", errors="ignore")).hexdigest()
        payload = {"md5": md5, "sections": [e.__dict__ for e in sections]}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[OK] Wrote section index with {len(sections)} sections to {out}")

    def lookup(self, query: str) -> str:
        q = query.strip()
        qn = _norm_compact(q)
        q_well = _extract_query_well(q)
        q_well_n = _norm_compact(q_well) if q_well else None

        # Heuristic: prefer headings that contain requested keyword + well if present
        keywords = ["summary", "introduction", "conclusion", "results", "discussion", "abstract"]
        ql = q.lower()
        q_kw = next((k for k in keywords if k in ql), None)

        best_i = None
        best_score = -1.0
        for i, e in enumerate(self._entries):
            score = 0.0
            if e.heading_norm == qn:
                score += 50.0
            if q_kw and q_kw in e.heading.lower():
                score += 10.0
            if q_well_n and q_well_n in _norm_compact(e.heading):
                score += 15.0
            # token overlap on compact strings
            overlap = sum(1 for t in re.findall(r"[a-z0-9]{3,}", qn) if t in e.heading_norm)
            score += min(10.0, overlap)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None or best_score < 10.0:
            return "[SECTION] No matching section heading found for query."

        e = self._entries[best_i]
        pages = ""
        if e.start_page is not None and e.end_page is not None:
            pages = f" (pages {e.start_page}-{e.end_page})"
        return (
            f"[SECTION] {e.heading}\n"
            f"Source: {e.source}{pages}\n\n"
            f"{e.text}"
        )

    def get_tool(self):
        @tool
        def lookup_section(query: str) -> str:
            """Deterministically find a report section by heading (avoids TOC dotted-leader matches)."""
            return self.lookup(query)

        return lookup_section





