"""
Structured lookup tool for Well_picks_Volve_v1.dat.

Instead of relying on vector similarity + chunking, we parse the .dat file into
rows (well, formation, top/base, MD/TVD/TVDSS, quality) and answer queries via
deterministic lookup.
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
class WellPickRow:
    well: str  # e.g. "NO 15/9-19 A"
    formation: str  # e.g. "Sleipner Fm."
    pick_type: str  # "Top" | "Base" | "-"
    md_m: Optional[float]
    tvd_m: Optional[float]
    tvdss_m: Optional[float]
    quality: Optional[str]  # e.g. "Not logged", "Eroded", etc.


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_well(s: str) -> str:
    """Normalize well string for matching across formats."""
    s = s.upper()
    s = s.replace("WELL", "").replace("NO", "")
    # Handle "A" suffix variations: "15/9-F-15 A" -> "15/9-F-15" for matching
    # But keep it in the normalized form for exact matches
    # Remove all non-alphanumeric except keep structure
    normalized = re.sub(r"[^0-9A-Z]+", "", s)
    # Try to match with and without trailing "A" for better matching
    return normalized


def _norm_form(s: str) -> str:
    """Normalize formation string for fuzzy matching."""
    s = s.upper()
    # Common variants
    s = s.replace("FORMATION", "")
    s = s.replace("FM.", "FM")
    s = s.replace("FM", "FM")  # keep token
    return re.sub(r"[^0-9A-Z]+", "", s)


def _extract_query_well(query: str) -> Optional[str]:
    """
    Extract well name from query using a permissive pattern.
    This is a fallback - fuzzy matching against stored names is preferred.
    """
    # Very permissive pattern: match "15/9-..." followed by any alphanumeric/slash/dash/space sequence
    # This handles: 15/9-F-15 A, 15/9-C-2 AH, 15/9-19A, etc.
    # Try to capture the full well name including suffixes
    patterns = [
        # Pattern 1: "Well NO 15/9-..." with any suffix (capture more aggressively)
        r"(?:Well\s+NO\s+)?(15[\s_/-]*9[\s_/-]*[A-Z0-9\s_/-]+?)(?:\s+formations|\s+formation|\s+in|\s+for|$|[^\w/\s-])",
        # Pattern 2: Just "15/9-..." at word boundary (more permissive)
        r"\b(15[\s_/-]*9[\s_/-]*[A-Z0-9\s_/-]+?)(?:\s+formations|\s+formation|\s+in|\s+for|$|[^\w/\s-])",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, query, flags=re.IGNORECASE)
        if m:
            well = m.group(1).strip()
            # Clean up: remove trailing non-alphanumeric except spaces and dashes
            well = re.sub(r"[^\w\s/-]+$", "", well).strip()
            # Normalize spacing but preserve structure
            well = re.sub(r"\s+", " ", well)
            # Don't truncate - keep the full match
            if len(well) >= 5:  # Minimum reasonable well name length
                logger.debug(f"[WELL_PICKS] _extract_query_well: Extracted '{well}' from query: '{query}'")
                return well
    
    logger.debug(f"[WELL_PICKS] _extract_query_well: No match for query: '{query}'")
    return None


def _extract_query_formation(query: str) -> Optional[str]:
    q = query.lower()
    # Try patterns: "depth of X", "depth of X formation", "depth of X fm"
    m = re.search(r"depth\s+of\s+(.+?)(?:\s+in\s+|$)", q)
    if m:
        cand = m.group(1).strip()
        cand = re.sub(r"\bformation\b", "", cand).strip()
        return cand
    # fallback: "formations in ..." doesn't need formation
    return None


class WellPicksTool:
    def __init__(
        self,
        dat_path: str,
        cache_path: str = "./data/well_picks_cache.json",
    ):
        self.dat_path = Path(dat_path)
        self.cache_path = Path(cache_path)
        self._rows: List[WellPickRow] = []
        self._by_well: Dict[str, List[WellPickRow]] = {}
        self._well_labels: List[Tuple[str, str]] = []  # (original_label, normalized_key) for fuzzy matching

        # Gracefully handle missing files - don't crash, just initialize empty
        try:
            self._load_or_parse()
        except FileNotFoundError:
            logger.info(
                f"[WELL_PICKS] Well picks data not available (dat: {self.dat_path}, cache: {self.cache_path}). Tool will return helpful messages. This is optional - app works without it."
            )
            # Initialize empty - tool will still work but return informative messages
            self._rows = []
            self._by_well = {}
            self._well_labels = []

    def _load_or_parse(self) -> None:
        # If the .dat file is missing, try loading from cache directly.
        if not self.dat_path.exists():
            if self.cache_path.exists():
                try:
                    payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    if isinstance(payload.get("rows"), list):
                        self._rows = [WellPickRow(**r) for r in payload["rows"]]
                        self._rebuild_index()
                        logger.info(f"[OK] Loaded well picks cache with {len(self._rows)} rows (dat missing)")
                        return
                except Exception as e:
                    logger.warning(f"[WELL_PICKS] Cache load failed while dat missing: {e}")
            raise FileNotFoundError(f"Well picks .dat not found and no cache available: {self.dat_path}")

        md5 = _file_md5(self.dat_path)
        if self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if payload.get("md5") == md5 and isinstance(payload.get("rows"), list):
                    self._rows = [WellPickRow(**r) for r in payload["rows"]]
                    self._rebuild_index()
                    logger.info(f"[OK] Loaded well picks cache with {len(self._rows)} rows")
                    return
            except Exception as e:
                logger.warning(f"[WELL_PICKS] Cache load failed, reparsing: {e}")

        self._rows = self._parse_dat(self.dat_path)
        self._rebuild_index()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache_path.write_text(
                json.dumps({"md5": md5, "rows": [asdict(r) for r in self._rows]}, indent=2),
                encoding="utf-8",
            )
            logger.info(f"[OK] Parsed well picks and wrote cache ({len(self._rows)} rows)")
        except Exception as e:
            logger.warning(f"[WELL_PICKS] Failed to write cache: {e}")

    def _rebuild_index(self) -> None:
        self._by_well = {}
        self._well_labels = []
        for r in self._rows:
            norm_key = _norm_well(r.well)
            self._by_well.setdefault(norm_key, []).append(r)
            # Build label list for fuzzy matching (avoid duplicates)
            if not any(label == r.well for label, _ in self._well_labels):
                self._well_labels.append((r.well, norm_key))

    def _parse_dat(self, path: Path) -> List[WellPickRow]:
        qlf_meanings = {
            "ER": "Eroded",
            "FP": "Faulted pick",
            "FO": "Faulted out",
            "NL": "Not logged",
            "NR": "Not reached",
        }

        rows: List[WellPickRow] = []
        current_well: Optional[str] = None
        in_data_section = False
        header_seen = False

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue

                if line.strip().startswith("Well NO ") and "Well name" not in line:
                    # Keep canonical like "NO 15/9-19 A" (strip "Well ")
                    current_well = line.strip()[5:].strip()
                    in_data_section = False
                    header_seen = False
                    continue

                if "Well name" in line and "Surface name" in line:
                    in_data_section = True
                    header_seen = True
                    continue

                if re.match(r"^[\s\-]+$", line):
                    continue

                if not (current_well and in_data_section and header_seen):
                    continue

                # Data row begins with "NO 15/9-.."
                if not re.match(r"^\s*NO\s+\d+[/\-]\d+[/\-].+", line):
                    continue

                # Example fixed-ish width, but we parse flexibly:
                # NO 15/9-19 A             Sleipner Fm. Top                         1         3919.59  3126.40  -3101.40 ...
                # Strategy: find obs# and MD; surface name is between well token and obs#
                well_match = re.match(r"^\s*(NO\s+.+?)\s{2,}(.+)$", line)
                if not well_match:
                    continue
                # The file repeats well in each row but it may include suffix "A" as separate token.
                # We'll trust current_well from the header for canonical mapping.
                remainder = well_match.group(2).strip()

                obs_match = re.search(r"(\d+)\s+([A-Z]{2,3}|\s{2,3})\s+([\d\.\-]+)", remainder)
                if not obs_match:
                    # fallback: split and look for first numeric as md after some text
                    parts = remainder.split()
                    # find first numeric token
                    num_start = None
                    for i, p in enumerate(parts):
                        if re.fullmatch(r"[\d\.\-]+", p):
                            num_start = i
                            break
                    if num_start is None:
                        continue
                    surface = " ".join(parts[:num_start]).strip()
                    nums = parts[num_start : num_start + 3]
                    md = float(nums[0]) if len(nums) >= 1 else None
                    tvd = float(nums[1]) if len(nums) >= 2 else None
                    tvdss = float(nums[2]) if len(nums) >= 3 else None
                    qlf = None
                else:
                    surface = remainder[: obs_match.start()].strip()
                    qlf_raw = obs_match.group(2).strip() or None
                    md = float(obs_match.group(3)) if obs_match.group(3) else None
                    after_md = remainder[obs_match.end() :].strip()
                    numbers = re.findall(r"[\d\.\-]+", after_md)
                    tvd = float(numbers[0]) if len(numbers) >= 1 else None
                    tvdss = float(numbers[1]) if len(numbers) >= 2 else None
                    qlf = qlf_meanings.get(qlf_raw, qlf_raw) if qlf_raw else None

                # Extract pick type from surface name
                pick_type = "-"
                formation = surface
                if formation.endswith(" Top"):
                    pick_type = "Top"
                    formation = formation[: -4].strip()
                elif formation.endswith(" Base"):
                    pick_type = "Base"
                    formation = formation[: -5].strip()

                rows.append(
                    WellPickRow(
                        well=current_well,
                        formation=formation,
                        pick_type=pick_type,
                        md_m=md,
                        tvd_m=tvd,
                        tvdss_m=tvdss,
                        quality=qlf or "Not logged",
                    )
                )

        return rows

    def _match_well_fuzzy(self, query_text: str) -> Optional[Tuple[str, str]]:
        """
        Use fuzzy matching against stored well names to find the best match.
        This is the state-of-the-art approach: data-driven matching instead of regex patterns.
        
        Returns (matched_well_label, normalized_key) or None.
        """
        if not self._well_labels:
            return None
        
        try:
            from rapidfuzz import process, fuzz
            
            # Extract potential well name from query using a very permissive pattern
            # This just finds something that looks like a well name
            potential_well = None
            well_pattern = re.search(
                r"(?:Well\s+NO\s+)?(15[\s_/-]*9[\s_/-]*[A-Z0-9\s_/-]+?)(?:\s|$|[^\w/])",
                query_text,
                re.IGNORECASE
            )
            if well_pattern:
                potential_well = well_pattern.group(1).strip()
                # Clean up
                potential_well = re.sub(r"[^\w\s/]+$", "", potential_well).strip()
                potential_well = re.sub(r"\s+", " ", potential_well)
            
            if not potential_well or len(potential_well) < 5:
                # Fallback: use a shorter substring if query is short
                if len(query_text) < 50:
                    potential_well = query_text.strip()
                else:
                    return None
            
            # Use RapidFuzz to find the best matching stored well name
            # This handles typos, spacing variations, case differences, format variations, etc.
            choices = [label for label, _ in self._well_labels]
            
            # Get top matches to prefer longer, more complete matches
            results = process.extract(
                potential_well,
                choices,
                scorer=fuzz.WRatio,  # Weighted ratio - best for well names with variations
                limit=5  # Get top 5 matches
            )
            
            if not results:
                return None
            
            # Prefer matches that are longer and more complete (contain more of the query)
            # This prevents "15/9-F-15 A" from matching "15/9-F-1" when "15/9-F-15 A" exists
            best_match = None
            best_score = 0
            for matched_label, score, _ in results:
                if score < 65:  # Minimum threshold
                    continue
                # Prefer matches that are longer (more complete) or have higher score
                # If scores are close (within 5%), prefer the longer match
                if best_match is None or score > best_score + 5 or (score >= best_score - 5 and len(matched_label) > len(best_match[0])):
                    best_match = (matched_label, score)
                    best_score = score
            
            if best_match:
                result = (best_match[0], best_score, None)
            else:
                return None
            
            if result:
                matched_label, score, _ = result
                # Find the normalized key for this matched label
                for label, norm_key in self._well_labels:
                    if label == matched_label:
                        logger.info(f"[WELL_PICKS] Fuzzy matched '{potential_well}' -> '{matched_label}' (score: {score:.1f}%)")
                        return (matched_label, norm_key)
            
            return None
        except Exception as e:
            logger.warning(f"[WELL_PICKS] Fuzzy matching failed: {e}")
            return None

    def lookup(self, query: str) -> str:
        logger.info(f"[WELL_PICKS] lookup called with query: '{query}'")
        # If no data loaded, return helpful message
        if not self._rows:
            return (
                "[WELL_PICKS] Well picks data not available. The .dat file and cache are missing. "
                "You can still query well information using the document retrieval tool."
            )
        
        # Try regex extraction first (for backwards compatibility and speed)
        well_q = _extract_query_well(query) or ""
        form_q = _extract_query_formation(query)
        norm_w = _norm_well(well_q) if well_q else None
        logger.info(f"[WELL_PICKS] Extracted well_q='{well_q}', norm_w='{norm_w}', form_q='{form_q}'")
        
        # If regex extraction failed or didn't produce a match, use fuzzy matching against stored well names
        # This is the data-driven approach: use actual stored data as source of truth
        if not norm_w or (norm_w and norm_w not in self._by_well):
            logger.info(f"[WELL_PICKS] Regex extraction failed or no exact match, trying fuzzy matching against {len(self._well_labels)} stored well names...")
            fuzzy_match = self._match_well_fuzzy(query)
            if fuzzy_match:
                matched_label, norm_w = fuzzy_match
                well_q = matched_label
                logger.info(f"[WELL_PICKS] Fuzzy match successful: '{matched_label}' (normalized: {norm_w})")
            elif not norm_w:
                # Both regex and fuzzy matching failed
                logger.warning(f"[WELL_PICKS] Both regex and fuzzy matching failed for query: '{query}'")

        ql = query.lower()

        # Complete list across ALL wells (deterministic, exhaustive)
        # BUT only if NO specific well is mentioned
        wants_all_wells = (
            ("formation" in ql or "formations" in ql)
            and ("well" in ql or "wells" in ql)
            and any(k in ql for k in ["each", "every", "all", "complete", "entire"])
            and not norm_w  # CRITICAL: Only want all wells if no specific well is detected
        )
        if wants_all_wells:
            all_wells: List[Tuple[str, List[str]]] = []
            for _nw, rows in self._by_well.items():
                if not rows:
                    continue
                well_label = rows[0].well
                formations = sorted({r.formation for r in rows})
                all_wells.append((well_label, formations))

            all_wells.sort(key=lambda x: x[0])
            lines = [f"[WELL_PICKS_ALL] Wells: {len(all_wells)}"]
            for well_label, formations in all_wells:
                lines.append(f"\n{well_label}:")
                for f in formations:
                    lines.append(f"- {f}")
            return "\n".join(lines)

        if not norm_w:
            return "[WELL_PICKS] No well detected in query. Provide a well like 15/9-19A."

        # Try exact match first
        candidates = self._by_well.get(norm_w, [])
        if candidates:
            logger.info(f"[WELL_PICKS] Exact match found for '{well_q}' (normalized: {norm_w})")
        
        # If no exact match, try fuzzy matching strategies
        if not candidates:
            # Strategy 1: Try without trailing "A" if query has "A" (e.g., "159F15A" -> "159F15")
            # This handles cases where stored well is "15/9-F-15" but query is "15/9-F-15 A"
            if norm_w.endswith("A"):
                norm_w_no_a = norm_w[:-1]  # Remove trailing "A"
                candidates = self._by_well.get(norm_w_no_a, [])
                if candidates:
                    logger.info(f"[WELL_PICKS] Matched well '{well_q}' by removing trailing 'A' (normalized: {norm_w_no_a})")
            
            # Strategy 2: Try adding "A" if query doesn't have it but stored does
            # This handles cases where stored well is "15/9-F-15 A" but query is "15/9-F-15"
            if not candidates and not norm_w.endswith("A"):
                norm_w_with_a = norm_w + "A"
                candidates = self._by_well.get(norm_w_with_a, [])
                if candidates:
                    logger.info(f"[WELL_PICKS] Matched well '{well_q}' by adding trailing 'A' (normalized: {norm_w_with_a})")
        
        # If still no match, try fuzzy matching on well names
        if not candidates:
            # Try multiple strategies:
            # 1. Check if normalized names are similar (prefix/suffix match)
            for stored_norm, stored_rows in self._by_well.items():
                # Check if the normalized well names are similar (e.g., "159F15" vs "159F15A")
                if stored_norm.startswith(norm_w) or norm_w.startswith(stored_norm):
                    # If one is a prefix of the other, they're likely the same well
                    if abs(len(stored_norm) - len(norm_w)) <= 2:  # Allow up to 2 char difference
                        candidates = stored_rows
                        logger.info(f"[WELL_PICKS] Matched well '{well_q}' via prefix match (stored: {stored_norm}, query: {norm_w})")
                        break
            
            # 2. If still no match, try matching on the original well labels (case-insensitive)
            if not candidates:
                well_q_upper = well_q.upper()
                for stored_norm, stored_rows in self._by_well.items():
                    if not stored_rows:
                        continue
                    stored_label = stored_rows[0].well.upper()
                    # Check if query well name appears in stored label or vice versa
                    if (well_q_upper in stored_label or stored_label in well_q_upper or
                        well_q_upper.replace(" ", "") == stored_label.replace(" ", "") or
                        _norm_well(well_q_upper) == stored_norm):
                        candidates = stored_rows
                        logger.info(f"[WELL_PICKS] Matched well '{well_q}' via label match (stored label: {stored_rows[0].well}, query: {well_q})")
                        break
        
        if not candidates:
            # Provide helpful error with available wells
            available = sorted([rows[0].well for rows in self._by_well.values() if rows])[:10]
            available_norm = sorted(list(self._by_well.keys()))[:10]
            logger.warning(f"[WELL_PICKS] No match found. Query well='{well_q}', normalized='{norm_w}'. Available normalized keys (sample): {available_norm}")
            return f"[WELL_PICKS] No rows found for well '{well_q}' (normalized: {norm_w}). Available wells (sample): {', '.join(available)}..."

        is_depth = "depth" in ql or "md" in ql or "tvd" in ql or "tvdss" in ql
        is_list_formations = ("list" in ql or "all" in ql) and "formation" in ql
        is_formations_in_well = "formations" in ql and ("in" in ql or "for" in ql or "all" in ql)

        if is_list_formations or is_formations_in_well:
            formations = sorted({r.formation for r in candidates})
            result = "\n".join([f"[WELL_PICKS] Well {candidates[0].well} formations:"] + [f"- {f}" for f in formations])
            logger.info(f"[WELL_PICKS] Returning formation list for well '{candidates[0].well}': {len(formations)} formations")
            return result

        if is_depth and form_q:
            nf = _norm_form(form_q)
            matched = [r for r in candidates if nf and nf in _norm_form(r.formation)]
            if not matched:
                # try substring fallback (e.g. "hugin" in "Hugin Fm. VOLVE")
                matched = [r for r in candidates if form_q.lower() in r.formation.lower()]

            if not matched:
                return f"[WELL_PICKS] Formation '{form_q}' not found in well {candidates[0].well}."

            # Summarize top/base if present
            top = next((r for r in matched if r.pick_type == "Top"), None)
            base = next((r for r in matched if r.pick_type == "Base"), None)

            def fmt_row(r: WellPickRow) -> str:
                parts = [f"{r.pick_type}:" if r.pick_type != "-" else "Pick:"]
                if r.md_m is not None:
                    parts.append(f"MD {r.md_m:.2f} m")
                if r.tvd_m is not None:
                    parts.append(f"TVD {r.tvd_m:.2f} m")
                if r.tvdss_m is not None:
                    parts.append(f"TVDSS {r.tvdss_m:.2f} m")
                return " ".join(parts)

            header = f"[WELL_PICKS] {matched[0].formation} in well {candidates[0].well}"
            lines = [header]
            if top:
                lines.append(f"- {fmt_row(top)}")
            if base:
                lines.append(f"- {fmt_row(base)}")
            if not top and not base:
                lines.extend([f"- {fmt_row(r)}" for r in matched[:10]])
            return "\n".join(lines)

        # Default: return a compact preview for the well (top 25 rows)
        lines = [f"[WELL_PICKS] Preview rows for well {candidates[0].well} (showing up to 25):"]
        for r in candidates[:25]:
            lines.append(
                f"- {r.formation} {r.pick_type}: MD={r.md_m}, TVD={r.tvd_m}, TVDSS={r.tvdss_m} (Q={r.quality})"
            )
        return "\n".join(lines)

    def get_tool(self):
        """Return a LangChain tool wrapping the structured lookup."""

        @tool
        def lookup_well_picks(query: str) -> str:
            """Structured lookup for formation/depth/interval picks from Well_picks_Volve_v1.dat."""
            return self.lookup(query)

        return lookup_well_picks


