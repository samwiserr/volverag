"""
StructuredFactsTool

Goal: provide deterministic lookup for numeric "facts" appearing in notes / narrative text
across all documents (e.g., Rw, temperature gradient, cutoffs, densities, constants).

We build an index from `section_index.json` (already produced during build-index).
This is intentionally conservative: it captures key/value-style statements and a few common
petrophysical note patterns, attaches well + source + page range for traceability.
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
class FactRow:
    well: str
    parameter: str
    value: str
    unit: Optional[str]
    source: str
    page_start: Optional[int]
    page_end: Optional[int]
    context: Optional[str]  # original line/snippet for auditing


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_well_key(s: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", s.upper())


def _canonicalize_well(s: str) -> str:
    w = s.strip().upper()
    w = w.replace("_", "/").replace(" ", "")
    w = w.replace("15-9", "15/9")
    w = w.replace("15/9F", "15/9-F")
    w = re.sub(r"(15/9-F)(\d)", r"\1-\2", w)
    w = w.replace("--", "-")
    return w


def _extract_well(text: str) -> Optional[str]:
    """Extract well name using centralized extraction function."""
    from ..normalize.query_normalizer import extract_well
    return extract_well(text)


def _extract_well_from_source_path(source: str) -> Optional[str]:
    # Try to infer from folder names like 15_9-F-5 or 15_9-F5
    m = re.search(r"(15[\s_/-]*9[\s_/-]*F[\s_/-]*-?\s*\d+[A-Z]?)", source, re.IGNORECASE)
    if not m:
        return None
    return _canonicalize_well(m.group(1))


def _norm_param(s: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", s.upper())


class StructuredFactsTool:
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Facts cache not found: {self.cache_path}")
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._rows: List[FactRow] = [FactRow(**r) for r in payload.get("rows", [])]
        self._by_well: Dict[str, List[FactRow]] = {}
        for r in self._rows:
            self._by_well.setdefault(_norm_well_key(r.well), []).append(r)
        logger.info(f"[OK] Loaded facts cache with {len(self._rows)} rows")

    @staticmethod
    def build_index(
        documents_root: str,
        out_path: str,
        section_index_path: Optional[str] = None,
    ) -> None:
        """
        Build a facts cache by scanning:
        - PDFs page-by-page under documents_root (primary, covers "notes" not captured as headings)
        - Optional section_index.json (secondary, adds structured section snippets)
        """
        root = Path(documents_root)
        if not root.exists():
            raise FileNotFoundError(f"Documents root not found: {root}")

        rows: List[FactRow] = []

        # Common noise keys to skip
        skip_key_norm = {
            "STATUS",
            "GRADERING",
            "UTLØPSDATO",
            "SIDE",
            "PAGE",
        }

        # Generic key/value patterns: "Key: 0.07 ohmm", "Key = 111 oC"
        kv_re = re.compile(
            r"^\s*(?P<key>[A-Za-z][A-Za-z0-9 _/\-\(\)\.\%]{0,60}?)\s*[:=]\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-zµ°/%]+(?:\s*[A-Za-zµ°/%]+)?)?\s*$",
            re.IGNORECASE,
        )

        # Special patterns seen often in these reports
        rw_re = re.compile(r"Rw\s*=\s*([-\d\.]+)\s*([A-Za-z]+)?\s*at\s*([-\d\.]+)\s*o?C", re.IGNORECASE)
        tempgrad_re = re.compile(r"Temp(?:erature)?\s*Gradient\s*[:=]\s*([-\d\.]+)\s*o?C", re.IGNORECASE)
        res_temp_re = re.compile(r"Reservo(?:i|a)r\s+Temperature\s*[:=]\s*([-\d\.]+)\s*o?C.*?(\d+)\s*m\s*TVDSS", re.IGNORECASE)

        def harvest_lines(
            well: str,
            source: str,
            page_start: Optional[int],
            page_end: Optional[int],
            text: str,
        ) -> None:
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if any(bad in line.lower() for bad in ["gradering:", "utløpsdato", "side "]):
                    continue

                # Special note patterns first
                m = rw_re.search(line)
                if m:
                    rows.append(
                        FactRow(
                            well=well,
                            parameter="Rw (at 20 °C)",
                            value=m.group(1),
                            unit=(m.group(2) or "ohmm"),
                            source=source,
                            page_start=page_start,
                            page_end=page_end,
                            context=line,
                        )
                    )
                    # Same line often also contains temp gradient; capture both.
                    mg = tempgrad_re.search(line)
                    if mg:
                        rows.append(
                            FactRow(
                                well=well,
                                parameter="Temperature gradient",
                                value=mg.group(1),
                                unit="°C",
                                source=source,
                                page_start=page_start,
                                page_end=page_end,
                                context=line,
                            )
                        )
                    continue

                m = tempgrad_re.search(line)
                if m:
                    rows.append(
                        FactRow(
                            well=well,
                            parameter="Temperature gradient",
                            value=m.group(1),
                            unit="°C",
                            source=source,
                            page_start=page_start,
                            page_end=page_end,
                            context=line,
                        )
                    )
                    continue

                m = res_temp_re.search(line)
                if m:
                    rows.append(
                        FactRow(
                            well=well,
                            parameter=f"Reservoir temperature (at {m.group(2)} m TVDSS)",
                            value=m.group(1),
                            unit="°C",
                            source=source,
                            page_start=page_start,
                            page_end=page_end,
                            context=line,
                        )
                    )
                    continue

                # Generic key/value
                km = kv_re.match(line)
                if km:
                    key = km.group("key").strip()
                    key_norm = _norm_param(key)
                    if key_norm in skip_key_norm:
                        continue
                    val = km.group("val").strip()
                    unit = (km.group("unit") or "").strip() or None
                    rows.append(
                        FactRow(
                            well=well,
                            parameter=key,
                            value=val,
                            unit=unit,
                            source=source,
                            page_start=page_start,
                            page_end=page_end,
                            context=line,
                        )
                    )

        # 1) Scan PDFs directly (primary)
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError("PyMuPDF (fitz) is required to build facts cache") from e

        pdfs = sorted(root.rglob("*.pdf")) + sorted(root.rglob("*.PDF"))
        keyword_filter = re.compile(r"\b(rw|gradient|reservoir temperature|temperature gradient|cutoff|cut-off|rho)\b", re.IGNORECASE)
        for pdf in pdfs:
            well = _extract_well_from_source_path(str(pdf))
            if not well:
                continue
            try:
                doc = fitz.open(pdf)
            except Exception:
                continue
            try:
                for i in range(doc.page_count):
                    txt = doc.load_page(i).get_text("text")
                    # cheap filter to avoid indexing noise
                    if not keyword_filter.search(txt):
                        continue
                    # 1-based pages for UX
                    harvest_lines(well, str(pdf), i + 1, i + 1, txt)
            finally:
                doc.close()

        # 2) Optionally scan section index (secondary)
        if section_index_path:
            section_path = Path(section_index_path)
            if section_path.exists():
                payload = json.loads(section_path.read_text(encoding="utf-8"))
                sections = payload.get("sections", [])
                for s in sections:
                    text = s.get("text") or ""
                    source = s.get("source") or ""
                    sp = s.get("start_page")
                    ep = s.get("end_page")
                    well = _extract_well_from_source_path(source) or _extract_well(text)
                    if not well:
                        continue
                    harvest_lines(
                        well,
                        source,
                        sp if isinstance(sp, int) else None,
                        ep if isinstance(ep, int) else None,
                        text,
                    )

        # Write cache
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # md5: prefer section index if present, otherwise write an empty marker
        md5 = _md5_file(Path(section_index_path)) if section_index_path and Path(section_index_path).exists() else ""
        out.write_text(
            json.dumps({"md5": md5, "rows": [asdict(r) for r in rows]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[OK] Wrote facts cache with {len(rows)} rows to {out}")

    def lookup(self, query: str) -> str:
        ql = query.lower()
        well = _extract_well(query)
        if not well:
            return "[FACTS_JSON] " + json.dumps(
                {"error": "no_well_detected", "message": "No well detected. Provide a well like 15/9-F-5."},
                ensure_ascii=False,
            )

        candidates = self._by_well.get(_norm_well_key(well), [])
        if not candidates:
            return "[FACTS_JSON] " + json.dumps(
                {"error": "no_facts_for_well", "well": well, "message": f"No structured facts found for well {well}."},
                ensure_ascii=False,
            )

        # Parameter match: pick the longest parameter name that appears in the query, or use synonyms.
        synonyms = {
            "rw": "RW",
            "temperature gradient": "TEMPERATUREGRADIENT",
            "reservoir temperature": "RESERVOIRTEMPERATURE",
        }
        qnorm = _norm_param(query)

        wanted_norm: Optional[str] = None
        for needle, kn in synonyms.items():
            if needle in ql:
                wanted_norm = kn
                break

        if wanted_norm is None:
            # Longest match among known parameter norms
            param_norms = sorted({_norm_param(r.parameter) for r in candidates}, key=len, reverse=True)
            for pn in param_norms:
                if pn and pn in qnorm:
                    wanted_norm = pn
                    break

        # If no parameter specified, return a small preview list (top 20 unique params)
        if wanted_norm is None:
            uniq: Dict[str, FactRow] = {}
            for r in candidates:
                uniq.setdefault(_norm_param(r.parameter), r)
            preview = list(uniq.values())[:20]
            return "[FACTS_JSON] " + json.dumps(
                {
                    "well": well,
                    "parameter": None,
                    "matches": [asdict(r) for r in preview],
                    "message": "No parameter detected; showing a preview of available parameters for this well.",
                },
                ensure_ascii=False,
            )

        matches = [r for r in candidates if _norm_param(r.parameter) == wanted_norm or wanted_norm in _norm_param(r.parameter)]
        # Sort deterministically by source then page
        matches.sort(key=lambda r: (r.source, r.page_start or -1, r.parameter))

        # De-duplicate identical matches (common when the same page is indexed from both PDF scan and section index)
        uniq: List[FactRow] = []
        seen: set[tuple] = set()
        for r in matches:
            k = (r.parameter, r.value, r.unit, r.source, r.page_start, r.page_end)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)

        return "[FACTS_JSON] " + json.dumps(
            {"well": well, "parameter_norm": wanted_norm, "matches": [asdict(r) for r in uniq]},
            ensure_ascii=False,
        )

    def get_tool(self):
        @tool
        def lookup_structured_facts(query: str) -> str:
            """Lookup numeric facts from notes/narrative text (Rw, gradients, temperatures, cutoffs, etc.) by well and parameter."""
            return self.lookup(query)

        return lookup_structured_facts


