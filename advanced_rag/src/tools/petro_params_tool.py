"""
Structured lookup tool for petrophysical parameter tables (Net/Gross, PHIF, SW, KLOGH...).

KLOGH = Klinkenberg-corrected horizontal permeability (arithmetic/harmonic/geometric means).

These values often appear in PDF tables. Even with strong hybrid retrieval, answers can fail because
tables may be fragmented or multiple similar tables exist across wells. This tool parses the extracted
table text into rows and answers deterministically.
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

from .well_picks_tool import _norm_well as _norm_well_picks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PetroParamRow:
    well: str  # e.g. "15/9-F-12"
    formation: str  # e.g. "Heather"
    netgros: Optional[float]
    phif: Optional[float]
    sw: Optional[float]
    klogh_a: Optional[float]
    klogh_h: Optional[float]
    klogh_g: Optional[float]
    source: str
    page_start: Optional[int]
    page_end: Optional[int]


def _md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_well(s: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", s.upper())


def _extract_well(text: str) -> Optional[str]:
    """Extract well name using centralized extraction function."""
    from ..normalize.query_normalizer import extract_well
    return extract_well(text)


def _safe_float(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except Exception:
        return None


class PetroParamsTool:
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Petro params cache not found: {self.cache_path}")
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._rows: List[PetroParamRow] = [PetroParamRow(**r) for r in payload.get("rows", [])]
        self._by_well: Dict[str, List[PetroParamRow]] = {}
        for r in self._rows:
            self._by_well.setdefault(_norm_well(r.well), []).append(r)
        logger.info(f"[OK] Loaded petro params cache with {len(self._rows)} rows")

    @staticmethod
    def build_index(section_index_path: str, out_path: str) -> None:
        section_path = Path(section_index_path)
        if not section_path.exists():
            raise FileNotFoundError(f"Section index not found: {section_path}")

        payload = json.loads(section_path.read_text(encoding="utf-8"))
        sections = payload.get("sections", [])
        rows: List[PetroParamRow] = []

        for s in sections:
            heading = (s.get("heading") or "").lower()
            text = s.get("text") or ""
            source = s.get("source") or ""
            start_page = s.get("start_page")
            end_page = s.get("end_page")

            # Identify candidate sections that likely contain the table
            if ("petrophysical parameters" not in text.lower()) and ("netgros" not in text.lower()):
                continue
            if "klogh" not in text.lower() and "netgros" not in text.lower():
                continue

            well = _extract_well(text) or _extract_well(source)
            if not well:
                continue

            # Restrict parsing to the actual table block (before CPI/next section),
            # otherwise we may accidentally parse CPI depth listings like "Heather 8".
            lower_text = text.lower()
            start_pos = lower_text.find("petrophysical parameters")
            if start_pos < 0:
                continue
            end_pos = len(text)
            # Prefer stopping at "*A:" footnote or "2.2" or "cpi"
            for marker in ["*a:", "\n2.2", "\n2.2 ", "\ncpi", "cpi "]:
                mp = lower_text.find(marker, start_pos)
                if mp != -1:
                    end_pos = min(end_pos, mp)
            table_text = text[start_pos:end_pos]

            # Tokenize while preserving decimals; split by whitespace
            tokens = [t for t in re.split(r"\s+", table_text.replace("\u00a0", " ")) if t]
            # Find header anchor "Formation" then "NetGros" then read rows after that
            # We'll locate the first occurrence of "Formation" and "NetGros" to start parsing.
            try:
                start_idx = next(i for i, t in enumerate(tokens) if t.lower().startswith("formation"))
            except StopIteration:
                continue

            # Advance to just after header block; headers can include multiple tokens
            # We skip until we hit the first formation name candidate followed by a number.
            i = start_idx + 1

            def is_num(t: str) -> bool:
                return re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t) is not None

            # formation names are typically alphabetic (Heather, Hugin, Sleipner, Skagerrak)
            def is_form_name(t: str) -> bool:
                return re.fullmatch(r"[A-Za-z][A-Za-z\.\-]*", t) is not None

            header_stop = {
                "formation", "netgros", "phif", "sw", "sw.", "petrophysical", "parameters",
                "aritmetic", "arithmetic", "harmonic", "geometric",
            }

            # Walk tokens and parse rows: <formation> <netgros> <phif> <sw> <klogh_a> <klogh_h> <klogh_g>
            # Some tables omit some klogh columns; we tolerate missing by stopping early.
            while i < len(tokens) - 3:
                t = tokens[i]
                if not is_form_name(t):
                    i += 1
                    continue
                tl = t.lower()
                if tl in header_stop or tl.startswith("klogh"):
                    i += 1
                    continue

                formation = t.strip().strip(":")
                # next numeric tokens
                nums: List[float] = []
                j = i + 1
                while j < len(tokens) and len(nums) < 6:
                    if is_num(tokens[j]):
                        nums.append(float(tokens[j]))
                    j += 1
                    # Stop if next formation starts and we already got at least 3 numbers (NetGros/Phif/Sw)
                    if len(nums) >= 3 and j < len(tokens) and is_form_name(tokens[j]):
                        break

                if len(nums) >= 3:
                    # Validate NetGros/Phif/Sw are fractions; if not, this isn't a real table row
                    if not (0.0 <= nums[0] <= 1.0 and 0.0 <= nums[1] <= 1.0 and 0.0 <= nums[2] <= 1.0):
                        i += 1
                        continue
                    # Pad to 6
                    while len(nums) < 6:
                        nums.append(None)  # type: ignore
                    row = PetroParamRow(
                        well=well,
                        formation=formation,
                        netgros=nums[0],
                        phif=nums[1],
                        sw=nums[2],
                        klogh_a=nums[3],
                        klogh_h=nums[4],
                        klogh_g=nums[5],
                        source=source,
                        page_start=start_page if isinstance(start_page, int) else None,
                        page_end=end_page if isinstance(end_page, int) else None,
                    )
                    rows.append(row)
                    i = j
                else:
                    i += 1

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        md5 = _md5_file(section_path)
        out.write_text(
            json.dumps({"md5": md5, "rows": [asdict(r) for r in rows]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[OK] Wrote petro params cache with {len(rows)} rows to {out}")

    def lookup(self, query: str) -> str:
        """
        Return a structured payload so the graph can render:
        - a single value (param + formation)
        - a parameter row (param only)
        - a formation column (formation only)
        - full table (well only)
        """
        logger.info(f"[PETRO_PARAMS] lookup() called with query: '{query}'")
        well = _extract_well(query)
        logger.info(f"[PETRO_PARAMS] Extracted well: '{well}'")
        if not well:
            logger.warning(f"[PETRO_PARAMS] No well detected in query: '{query}'")
            return "[PETRO_PARAMS_JSON] " + json.dumps(
                {"error": "no_well_detected", "message": "No well detected. Provide a well like 15/9-F-12."},
                ensure_ascii=False,
            )

        # Multi-strategy well matching (same approach as FormationPropertiesTool)
        nwell = _norm_well(well)
        rows = self._by_well.get(nwell, [])

        if not rows:
            # Try normalizing like well picks (remove NO/WELL etc)
            try:
                nw2 = _norm_well_picks(well)
            except Exception:
                nw2 = None
            if nw2:
                rows = self._by_well.get(nw2, [])

        if not rows:
            # Try extracting just the numeric/cleaned part
            cleaned = well.upper().replace("WELL", "").replace("NO", "").strip()
            nw3 = _norm_well(cleaned)
            rows = self._by_well.get(nw3, [])
        
        # Try matching with suffixes stripped from cache keys
        # Well names in cache may have suffixes like "PETROPHYSICAL", "DATO", "FORMATION"
        if not rows:
            common_suffixes = ["PETROPHYSICAL", "DATO", "FORMATION", "REPORT"]
            query_base = nwell
            logger.debug(f"[PETRO_PARAMS] Trying suffix-stripped matching for '{well}' (norm: '{nwell}')")
            
            for stored_norm, stored_rows in self._by_well.items():
                if not stored_rows:
                    continue
                # Strip suffixes from stored key
                stored_base = stored_norm
                for suffix in common_suffixes:
                    if stored_base.endswith(suffix):
                        stored_base = stored_base[:-len(suffix)]
                        logger.debug(f"[PETRO_PARAMS] Stripped suffix '{suffix}' from '{stored_norm}' -> '{stored_base}'")
                        break
                # Match if bases are equal
                if query_base == stored_base:
                    logger.info(f"[PETRO_PARAMS] ✅ Found match (suffix stripped): '{well}' (base: '{query_base}') -> '{stored_norm}' (base: '{stored_base}')")
                    rows = stored_rows
                    break

        if not rows:
            # Try matching against all stored well keys using normalized comparisons
            # Use exact match only - avoid substring matching which can cause false positives
            query_norm_clean = _norm_well(cleaned if 'cleaned' in locals() else well)
            query_norm_picks = None
            try:
                query_norm_picks = _norm_well_picks(well)
            except Exception:
                query_norm_picks = None

            logger.debug(f"[PETRO_PARAMS] Trying fuzzy matching for well '{well}' (norm_clean='{query_norm_clean}', norm_picks='{query_norm_picks}')")
            
            # First pass: exact matches only (including matches with common suffixes stripped)
            # Well names in cache may have suffixes like "PETROPHYSICAL", "DATO", "FORMATION"
            common_suffixes = ["PETROPHYSICAL", "DATO", "FORMATION", "REPORT"]
            query_well_base = query_norm_clean
            # Try to strip common suffixes from query (though it shouldn't have them)
            for suffix in common_suffixes:
                if query_well_base.endswith(suffix):
                    query_well_base = query_well_base[:-len(suffix)]
                    break
            
            for stored_norm, stored_rows in self._by_well.items():
                if not stored_rows:
                    continue
                # Try exact match first
                if query_norm_clean == stored_norm or (query_norm_picks and query_norm_picks == stored_norm):
                    logger.info(f"[PETRO_PARAMS] Found exact match: '{well}' -> '{stored_norm}'")
                    rows = stored_rows
                    break
                # Try matching with suffixes stripped from stored key
                stored_base = stored_norm
                for suffix in common_suffixes:
                    if stored_base.endswith(suffix):
                        stored_base = stored_base[:-len(suffix)]
                        break
                if query_well_base == stored_base or (query_norm_picks and query_norm_picks == stored_base):
                    logger.info(f"[PETRO_PARAMS] Found match (suffixes stripped): '{well}' (base: '{query_well_base}') -> '{stored_norm}' (base: '{stored_base}')")
                    rows = stored_rows
                    break
            
            # Second pass: only if no exact match, try very strict substring matching
            # Only match if one is a clear prefix/suffix of the other (e.g., "159F5" vs "159F5A")
            # CRITICAL: Must match the full well number, not just prefix (e.g., "159F5" vs "159F15A")
            if not rows:
                for stored_norm, stored_rows in self._by_well.items():
                    if not stored_rows:
                        continue
                    # Only match if one well is clearly a variant of the other
                    # e.g., "159F5" should match "159F5A" but NOT "159F15D"
                    if query_norm_clean and stored_norm:
                        # Extract the well number part (everything before the last letter/suffix)
                        # For "159F5", well_num = "159F5"
                        # For "159F15A", well_num = "159F15"
                        # For "159F5A", well_num = "159F5"
                        def extract_well_number(norm_str):
                            # Extract the well number part before any suffix letters
                            # Pattern: digits, optional letter (F), digits, then optional suffix letters
                            # e.g., "159F5" -> "159F5", "159F5A" -> "159F5", "159F15A" -> "159F15"
                            import re
                            # Match: start with digits, optional F, more digits, then optional letters at end
                            # Group 1 captures the numeric part (before suffix)
                            match = re.match(r'^(\d+[A-Z]?\d+)([A-Z]*)$', norm_str)
                            if match:
                                well_num = match.group(1)
                                logger.debug(f"[PETRO_PARAMS] extract_well_number: '{norm_str}' -> '{well_num}'")
                                return well_num
                            # Fallback: remove trailing letters
                            well_num = re.sub(r'[A-Z]+$', '', norm_str)
                            logger.debug(f"[PETRO_PARAMS] extract_well_number (fallback): '{norm_str}' -> '{well_num}'")
                            return well_num
                        
                        query_well_num = extract_well_number(query_norm_clean)
                        stored_well_num = extract_well_number(stored_norm)
                        
                        logger.debug(f"[PETRO_PARAMS] Comparing well numbers: query='{query_well_num}' vs stored='{stored_well_num}'")
                        
                        # Only match if well numbers are the same (allowing suffix differences)
                        # e.g., "159F5" matches "159F5A" but "159F5" does NOT match "159F15A"
                        if query_well_num == stored_well_num:
                            # Check length difference is small (only suffix difference)
                            if abs(len(query_norm_clean) - len(stored_norm)) <= 2:
                                logger.info(f"[PETRO_PARAMS] ✅ Found well number match: '{well}' (norm: '{query_norm_clean}', well_num: '{query_well_num}') -> '{stored_norm}' (well_num: '{stored_well_num}')")
                                rows = stored_rows
                                break
                        else:
                            logger.debug(f"[PETRO_PARAMS] ❌ Well numbers don't match: '{query_well_num}' != '{stored_well_num}'")

        if not rows:
            # Log available wells for debugging (sample)
            available_wells = sorted([r.well for rows_list in self._by_well.values() for r in (rows_list[:1] if rows_list else [])])[:10]
            all_normalized_wells = sorted(list(self._by_well.keys()))[:10]
            logger.warning(f"[PETRO_PARAMS] ❌ No rows found for well '{well}' (normalized: '{nwell}')")
            logger.warning(f"[PETRO_PARAMS] Available wells (sample): {available_wells}")
            logger.warning(f"[PETRO_PARAMS] Available normalized well keys (sample): {all_normalized_wells}")
            return "[PETRO_PARAMS_JSON] " + json.dumps(
                {"error": "no_rows_for_well", "well": well, "message": f"No petrophysical parameter rows found for well {well}."},
                ensure_ascii=False,
            )

        # Build a formation->values mapping. Values remain raw numbers (no interpretation).
        formations = sorted({r.formation for r in rows})
        by_form: Dict[str, Dict[str, Optional[float]]] = {}
        sources: Dict[Tuple[str, Optional[int], Optional[int]], None] = {}

        for r in rows:
            by_form[r.formation] = {
                "netgros": r.netgros,
                "phif": r.phif,
                "sw": r.sw,
                "klogh_a": r.klogh_a,
                "klogh_h": r.klogh_h,
                "klogh_g": r.klogh_g,
            }
            sources[(r.source, r.page_start, r.page_end)] = None

        payload = {
            "well": well,
            "formations": formations,
            "values": by_form,
            "sources": [{"source": s, "page_start": ps, "page_end": pe} for (s, ps, pe) in sources.keys()],
        }
        return "[PETRO_PARAMS_JSON] " + json.dumps(payload, ensure_ascii=False)

    def get_tool(self):
        @tool
        def lookup_petrophysical_params(query: str) -> str:
            """Structured lookup for Petrophysical Parameters table values (Net/Gross, PHIF, SW, KLOGH...)."""
            return self.lookup(query)

        return lookup_petrophysical_params


