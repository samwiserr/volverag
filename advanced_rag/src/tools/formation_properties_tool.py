"""
One-shot deterministic tool:

Given a query like:
  "complete list formations present in 15/9-F-4 and their petrophysical properties"

Return a single table:
- Formations: from Well_picks_Volve_v1.dat (authoritative list)
- Petrophysical properties: from parsed Petrophysical Parameters tables cache (if available)

If a formation has no parsed petrophysical row, we return N/A for that formation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.tools import tool

from .well_picks_tool import WellPicksTool, WellPickRow, _norm_well as _norm_well_picks
from .petro_params_tool import PetroParamsTool, PetroParamRow, _norm_well as _norm_well_petro

logger = logging.getLogger(__name__)


def _extract_platform_or_well(query: str) -> Optional[str]:
    """
    Extract a well token including platform-style wells like 15/9-F-4.
    Important: avoid accidentally capturing the "a" in "... 15/9-F-4 and ...".
    """
    m = re.search(
        r"(15[\s_/-]*9[\s_/-]*(?:F[\s_/-]*-?[\s_/-]*)?-?\s*\d+(?:\s*[A-Z])?\b)",
        query,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    w = m.group(1).strip()
    w = re.sub(r"\s+T\d+\b", "", w, flags=re.IGNORECASE).strip()
    return w


def _norm_form(s: str) -> str:
    s = s.upper()
    s = s.replace("FORMATION", "")
    s = s.replace("FM.", "FM")
    s = s.replace("FM", "")
    return re.sub(r"[^0-9A-Z]+", "", s)


def _fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


class FormationPropertiesTool:
    def __init__(
        self,
        well_picks_dat_path: str,
        petro_params_cache_path: str,
        well_picks_cache_path: str = "./data/well_picks_cache.json",
    ):
        self._well_picks = WellPicksTool(dat_path=well_picks_dat_path, cache_path=well_picks_cache_path)
        self._petro = PetroParamsTool(cache_path=petro_params_cache_path)

    def _get_formations_for_well(self, well_query: str) -> Tuple[str, List[str]]:
        nw = _norm_well_picks(well_query)
        rows = self._well_picks._by_well.get(nw, [])  # internal index (stable for our app)
        if not rows:
            raise KeyError(well_query)
        well_label = rows[0].well
        formations = sorted({r.formation for r in rows})
        return well_label, formations

    def _petro_rows_for_well(self, well_query: str) -> List[PetroParamRow]:
        """Get petro params for a well, trying multiple normalization approaches."""
        # Try direct normalization first
        nw = _norm_well_petro(well_query)
        rows = self._petro._by_well.get(nw, [])
        if rows:
            return rows
        
        # If not found, try normalizing like well picks (remove "NO", "WELL")
        nw2 = _norm_well_picks(well_query)
        rows = self._petro._by_well.get(nw2, [])
        if rows:
            return rows
        
        # Try extracting just the numeric part (e.g., "15/9-F-5" from "NO 15/9-F-5")
        # Remove "NO", "WELL" and normalize
        cleaned = well_query.upper().replace("WELL", "").replace("NO", "").strip()
        nw3 = _norm_well_petro(cleaned)
        rows = self._petro._by_well.get(nw3, [])
        if rows:
            return rows
        
        # Try matching against all stored well keys by comparing original well names
        # Well picks might have "NO 15/9-F-5" while petro params have "15/9-F-5"
        # The petro params cache is indexed by normalized keys, so we need to compare normalized forms
        query_norm_clean = _norm_well_petro(cleaned)  # Normalize cleaned query (e.g., "15/9-F-5" -> "159F5")
        query_norm_picks = _norm_well_picks(well_query)  # Normalize like well picks (e.g., "NO 15/9-F-5" -> "159F5")
        
        for stored_norm, stored_rows in self._petro._by_well.items():
            if not stored_rows:
                continue
            
            # stored_norm is already normalized (it's the key in _by_well dict)
            # Check if our normalized query matches the stored normalized key
            if query_norm_clean == stored_norm or query_norm_picks == stored_norm:
                return stored_rows
            
            # Also try substring matching
            if query_norm_clean in stored_norm or stored_norm in query_norm_clean:
                return stored_rows
            if query_norm_picks in stored_norm or stored_norm in query_norm_picks:
                return stored_rows
        
        return []

    def lookup(self, query: str) -> str:
        ql = query.lower()
        
        # Check if this is an "all wells" query - improved detection
        # Match patterns like:
        # - "list all well formations and their properties"
        # - "all formations and their properties"
        # - "all wells formations and properties"
        # - "each well formations and properties"
        # - "list all formations and their properties"
        
        # First, check if a specific well is mentioned (this takes priority)
        well = _extract_platform_or_well(query)
        
        # If no specific well detected, check for "all wells" patterns
        wants_all_wells = (
            not well  # No specific well detected
            and ("formation" in ql or "formations" in ql)
            and ("properties" in ql or "petrophysical" in ql)
            and any(k in ql for k in ["all", "each", "every", "complete", "entire", "list all"])
        )
        
        # Also handle "list all well formations" pattern
        if not wants_all_wells and not well:
            wants_all_wells = (
                ("list" in ql and "all" in ql)
                and ("formation" in ql or "formations" in ql)
                and ("properties" in ql or "petrophysical" in ql)
            )
        
        if wants_all_wells:
            # Return formations and properties for ALL wells
            logger.info(f"[FORMATION_PROPERTIES] Detected 'all wells' query: '{query}' - routing to _lookup_all_wells()")
            result = self._lookup_all_wells()
            logger.info(f"[FORMATION_PROPERTIES] _lookup_all_wells() returned {len(result)} characters")
            return result
        
        # If we have a specific well, use single-well lookup
        if not well:
            logger.warning(f"[FORMATION_PROPERTIES] No well detected in query: '{query}'")
            return "[WELL_FORMATION_PROPERTIES] No well detected. Provide a well like 15/9-F-4, or ask for 'all formations and their properties' to see all wells."

        try:
            well_label, formations = self._get_formations_for_well(well)
        except KeyError:
            return f"[WELL_FORMATION_PROPERTIES] No formation picks found for well '{well}'."

        petro_rows = self._petro_rows_for_well(well)
        petro_by_form: Dict[str, PetroParamRow] = {}
        for r in petro_rows:
            petro_by_form[_norm_form(r.formation)] = r

        lines: List[str] = []
        lines.append(f"[WELL_FORMATION_PROPERTIES] Well: {well_label}")
        lines.append(f"Formations (from well picks): {len(formations)}")
        lines.append("")
        lines.append("| Formation | Net/Gross | PHIF (Porosity) | SW | KLOGH (A/H/G) |")
        lines.append("|---|---:|---:|---:|---|")

        missing = 0
        used_sources: Dict[Tuple[str, Optional[int], Optional[int]], None] = {}

        for f in formations:
            nf = _norm_form(f)
            pr = petro_by_form.get(nf)

            if pr is None:
                # Try startswith-like matching (e.g., "Sleipner Fm." vs "Sleipner")
                pr = next(
                    (r for r in petro_rows if _norm_form(f).startswith(_norm_form(r.formation)) or _norm_form(r.formation).startswith(_norm_form(f))),
                    None,
                )

            if pr is None:
                missing += 1
                lines.append(f"| {f} | N/A | N/A | N/A | N/A |")
                continue

            used_sources[(pr.source, pr.page_start, pr.page_end)] = None
            klogh = f"{_fmt_num(pr.klogh_a, 3)}/{_fmt_num(pr.klogh_h, 3)}/{_fmt_num(pr.klogh_g, 3)}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f,
                        _fmt_num(pr.netgros, 3),
                        _fmt_num(pr.phif, 3),
                        _fmt_num(pr.sw, 3),
                        klogh,
                    ]
                )
                + " |"
            )

        lines.append("")
        lines.append("Notes:")
        lines.append("- Values come from parsed **Petrophysical Parameters** tables when available.")
        lines.append("- `N/A` means that formation does not have a parsed petrophysical-parameters row for this well (common/expected).")
        lines.append("")

        # Emit sources as separate lines so the Streamlit UI can make them clickable.
        if used_sources:
            lines.append("Sources:")
            for (src, ps, pe) in sorted(used_sources.keys(), key=lambda x: (x[0] or "", x[1] or -1, x[2] or -1)):
                pages = ""
                if ps is not None and pe is not None:
                    pages = f" (pages {ps}-{pe})"
                lines.append(f"Source: {src}{pages}")

        lines.append("")
        lines.append(f"Coverage: {len(formations) - missing}/{len(formations)} formations have petrophysical table values.")

        return "\n".join(lines)
    
    def _lookup_all_wells(self) -> str:
        """Return formations and properties for all wells."""
        logger.info(f"[FORMATION_PROPERTIES] _lookup_all_wells() called - processing all wells")
        lines: List[str] = []
        lines.append("[WELL_FORMATION_PROPERTIES] All Wells")
        lines.append("")
        
        all_wells_data: List[Tuple[str, List[str], Dict[str, PetroParamRow], List[PetroParamRow]]] = []
        
        # Get all wells from well picks
        for norm_w, rows in self._well_picks._by_well.items():
            if not rows:
                continue
            well_label = rows[0].well
            formations = sorted({r.formation for r in rows})
            
            # Get petro params for this well - try multiple approaches
            # The well_label from well picks is like "NO 15/9-F-5", but petro params has "15/9-F-5"
            petro_rows = self._petro_rows_for_well(well_label)
            
            # Debug: Log if we found petro params for this well
            if petro_rows:
                logger.debug(f"[FORMATION_PROPERTIES] Found {len(petro_rows)} petro param rows for well '{well_label}'")
            else:
                logger.debug(f"[FORMATION_PROPERTIES] No petro param rows found for well '{well_label}'")
            
            petro_by_form: Dict[str, PetroParamRow] = {}
            for r in petro_rows:
                petro_by_form[_norm_form(r.formation)] = r
            
            # Store petro_rows along with the data so we can use them for fuzzy matching
            all_wells_data.append((well_label, formations, petro_by_form, petro_rows))
        
        all_wells_data.sort(key=lambda x: x[0])
        
        for well_label, formations, petro_by_form, petro_rows in all_wells_data:
            lines.append(f"## {well_label}")
            lines.append("")
            lines.append("| Formation | Net/Gross | PHIF (Porosity) | SW | KLOGH (A/H/G) |")
            lines.append("|---|---:|---:|---:|---|")
            
            for f in formations:
                nf = _norm_form(f)
                pr = petro_by_form.get(nf)
                
                if pr is None:
                    # Try fuzzy matching - check if any petro param formation matches
                    # This handles cases like "Hugin Fm. VOLVE" vs "Hugin"
                    # We need to check against the actual petro_rows, not petro_by_form.values()
                    # because petro_by_form is keyed by normalized formation names
                    for petro_row in petro_rows:
                        petro_form_norm = _norm_form(petro_row.formation)
                        # Check if normalized forms match or one contains the other
                        # "HUGINVOLVE" should match "HUGIN" (well picks formation contains petro param formation)
                        if nf == petro_form_norm or nf.startswith(petro_form_norm) or petro_form_norm.startswith(nf):
                            pr = petro_row
                            break
                
                if pr is None:
                    lines.append(f"| {f} | N/A | N/A | N/A | N/A |")
                else:
                    klogh = f"{_fmt_num(pr.klogh_a, 3)}/{_fmt_num(pr.klogh_h, 3)}/{_fmt_num(pr.klogh_g, 3)}"
                    lines.append(
                        "| " + " | ".join([
                            f,
                            _fmt_num(pr.netgros, 3),
                            _fmt_num(pr.phif, 3),
                            _fmt_num(pr.sw, 3),
                            klogh,
                        ]) + " |"
                    )
            lines.append("")
        
        lines.append("Notes:")
        lines.append("- Values come from parsed **Petrophysical Parameters** tables when available.")
        lines.append("- `N/A` means that formation does not have a parsed petrophysical-parameters row for that well.")
        
        result = "\n".join(lines)
        logger.info(f"[FORMATION_PROPERTIES] _lookup_all_wells() completed - processed {len(all_wells_data)} wells, result length: {len(result)}")
        return result

    def get_tool(self):
        @tool
        def lookup_formation_properties(query: str) -> str:
            """One-shot: list formations for a well and their petrophysical properties (Net/Gross, PHIF, SW, KLOGH) when available.
            
            KLOGH = Klinkenberg-corrected horizontal permeability (A/H/G = Arithmetic/Harmonic/Geometric means).
            """
            return self.lookup(query)

        return lookup_formation_properties


