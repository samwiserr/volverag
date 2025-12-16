"""
Deterministic lookup tool for "Evaluation Parameters" tables.

These tables typically contain per-formation constants used in evaluation (e.g. Rhoma, Rhofl, Archie a/n/m,
GRmin/GRmax, etc.) and sometimes include assumptions like Rw and temperature gradient.

We parse these directly from PDFs (text extraction via PyMuPDF) during index build into a cache, then answer
queries deterministically with citations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.tools import tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalParamsTable:
    well: str  # canonical, e.g. "15/9-F-5"
    formations: List[str]  # e.g. ["Draupne", "Heather", "Hugin", "Sleipner"]
    params: Dict[str, Dict[str, str]]  # param -> formation -> value (string to preserve "*" etc.)
    notes: List[str]  # footnotes and assumptions like Rw/Temp gradient
    source: str
    page: int


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_well_key(s: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", s.upper())


def _canonicalize_well(s: str) -> str:
    # Normalize variations: 15_9-F5, 15/9-F5, 15/9-F-5
    w = s.strip().upper()
    w = w.replace("_", "/").replace(" ", "")
    w = w.replace("15-9", "15/9")
    # Ensure dash after F when missing
    w = re.sub(r"(15/9-F)(\d)", r"\1-\2", w)
    # Collapse doubles
    w = w.replace("--", "-")
    return w


def _extract_well(text: str) -> Optional[str]:
    """Extract well name using centralized extraction function."""
    from ..normalize.query_normalizer import extract_well
    return extract_well(text)


def _parse_eval_params_page(text: str) -> Optional[Tuple[str, List[str], Dict[str, Dict[str, str]], List[str]]]:
    """
    Parse a single page text containing:
      "Evaluation Parameters  15/9-F5"
      "Parameter Draupne Heather Hugin"
      "Sleipner"
      "Rhoma" then values...
    """
    if not re.search(r"evaluation\s+parameters", text, re.IGNORECASE):
        return None

    # Keep only the block starting at "Evaluation Parameters"
    start = re.search(r"Evaluation\s+Parameters", text, re.IGNORECASE)
    if not start:
        return None
    block = text[start.start() :]

    well = _extract_well(block) or _extract_well(text)
    if not well:
        return None

    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    # Find header line containing "Parameter"
    try:
        hidx = next(i for i, ln in enumerate(lines) if re.match(r"^Parameter\b", ln, re.IGNORECASE))
    except StopIteration:
        return None

    known_params = {"rhoma", "rhofl", "a", "b", "grmin", "grmax", "n", "m"}

    # Collect formations tokens after "Parameter", possibly spanning multiple lines until first param row
    formations: List[str] = []
    header_tokens = re.split(r"\s+", lines[hidx])
    # first token is "Parameter"
    formations.extend([t for t in header_tokens[1:] if t])
    j = hidx + 1
    while j < len(lines):
        tok0 = lines[j].split()[0].lower()
        if tok0 in known_params:
            break
        # If line looks like a single formation name (e.g. "Sleipner") add it
        if re.fullmatch(r"[A-Za-z][A-Za-z\-\.]*", lines[j]):
            formations.append(lines[j])
        else:
            # Also allow a split like "Hugin" and next line "Sleipner"
            formations.extend([t for t in re.split(r"\s+", lines[j]) if re.fullmatch(r"[A-Za-z][A-Za-z\-\.]*", t)])
        j += 1

    formations = [f.strip().strip(":") for f in formations if f.strip()]
    # Deduplicate while preserving order
    seen = set()
    formations = [f for f in formations if not (f.lower() in seen or seen.add(f.lower()))]
    if len(formations) < 2:
        return None

    params: Dict[str, Dict[str, str]] = {}
    notes: List[str] = []

    i = j
    # Parse parameter rows: <param> then N values (each may be on its own line)
    while i < len(lines):
        ln = lines[i]
        low = ln.split()[0].lower()

        # Notes/assumptions section
        if low.startswith("*") or low.startswith("rw") or "temp" in low or "reserv" in low:
            notes.append(ln)
            i += 1
            # Also capture trailing note lines
            while i < len(lines) and (lines[i].startswith("*") or lines[i].lower().startswith("rw") or "temp" in lines[i].lower() or "reserv" in lines[i].lower()):
                notes.append(lines[i])
                i += 1
            continue

        if low not in known_params:
            i += 1
            continue

        param_name = ln.split()[0]
        values: List[str] = []
        k = i + 1
        while k < len(lines) and len(values) < len(formations):
            # stop if next parameter starts
            nxt0 = lines[k].split()[0].lower()
            if nxt0 in known_params:
                break
            # Capture first token if it's a value
            tok = lines[k].split()[0]
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", tok) or tok == "*":
                values.append(tok)
            k += 1

        if len(values) != len(formations):
            # not parseable; abort
            return None

        params[param_name] = {formations[idx]: values[idx] for idx in range(len(formations))}
        i = k

    if not params:
        return None

    return well, formations, params, notes


class EvalParamsTool:
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Eval params cache not found: {self.cache_path}")
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._rows: List[EvalParamsTable] = [EvalParamsTable(**r) for r in payload.get("rows", [])]
        self._by_well: Dict[str, List[EvalParamsTable]] = {}
        for r in self._rows:
            self._by_well.setdefault(_norm_well_key(r.well), []).append(r)
        logger.info(f"[OK] Loaded eval params cache with {len(self._rows)} tables")

    @staticmethod
    def build_index(documents_root: str, out_path: str) -> None:
        """
        Scan PDFs under documents_root for 'Evaluation Parameters' pages and cache parsed tables.
        """
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError("PyMuPDF (fitz) is required to build eval params index") from e

        root = Path(documents_root)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(root.rglob("*.pdf")) + sorted(root.rglob("*.PDF"))
        # Prefer petrophysical report PDFs first (smaller subset)
        preferred = [p for p in pdfs if "PETROPHYSICAL_REPORT" in p.name.upper()]
        others = [p for p in pdfs if p not in preferred]
        scan_list = preferred + others

        rows: List[EvalParamsTable] = []
        md5s: Dict[str, str] = {}

        for pdf in scan_list:
            try:
                md5s[str(pdf)] = _md5_file(pdf)
            except Exception:
                md5s[str(pdf)] = ""
            try:
                doc = fitz.open(pdf)
            except Exception:
                continue

            try:
                for page_idx in range(doc.page_count):
                    text = doc.load_page(page_idx).get_text("text")
                    if not re.search(r"evaluation\s+parameters", text, re.IGNORECASE):
                        continue
                    parsed = _parse_eval_params_page(text)
                    if not parsed:
                        continue
                    well, formations, params, notes = parsed
                    rows.append(
                        EvalParamsTable(
                            well=well,
                            formations=formations,
                            params=params,
                            notes=notes,
                            source=str(pdf),
                            page=page_idx,
                        )
                    )
            finally:
                doc.close()

        out.write_text(
            json.dumps(
                {"md5s": md5s, "rows": [asdict(r) for r in rows]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"[OK] Wrote eval params cache with {len(rows)} tables to {out}")

    def lookup(self, query: str) -> str:
        ql = query.lower()
        well = _extract_well(query)
        if not well:
            return "[EVAL_PARAMS_JSON] " + json.dumps(
                {"error": "no_well_detected", "message": "No well detected. Provide a well like 15/9-F-5."},
                ensure_ascii=False,
            )

        tables = self._by_well.get(_norm_well_key(well), [])
        if not tables:
            return "[EVAL_PARAMS_JSON] " + json.dumps(
                {"error": "no_table_for_well", "well": well, "message": f"No evaluation-parameters table found for well {well}."},
                ensure_ascii=False,
            )

        # Take first table (usually unique per well)
        t = tables[0]

        payload = {
            "well": t.well,
            "formations": t.formations,
            "params": t.params,  # raw strings (may include "*")
            "notes": t.notes,  # raw strings (may include misspellings)
            "source": t.source,
            # Page in cache is 0-based; expose 1-based for humans/UX
            "page_start": t.page + 1,
            "page_end": t.page + 1,
        }
        return "[EVAL_PARAMS_JSON] " + json.dumps(payload, ensure_ascii=False)

    def get_tool(self):
        @tool
        def lookup_evaluation_parameters(query: str) -> str:
            """Lookup evaluation parameters table (Rhoma/Rhofl/GRmin/GRmax/Archie a,n,m, etc.) for a given well."""
            return self.lookup(query)

        return lookup_evaluation_parameters


