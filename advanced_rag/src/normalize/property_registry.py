"""
Property registry (data-driven).

Purpose:
- Provide a canonical set of properties that the system can answer deterministically
  from structured tools (eval params, petro params, facts).
- Provide robust matching via canonical tokenization (not endless hand synonyms).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class PropertyEntry:
    canonical: str  # e.g. "matrix_density"
    tool: str  # e.g. "lookup_evaluation_parameters"
    # Optional sub-key for tool formatting (e.g. eval params key "Rhoma")
    structured_key: Optional[str] = None
    # Display name for clarification menus
    display: Optional[str] = None


def canonical_tokenize(text: str) -> str:
    """
    Canonical tokenization for robust matching:
    - lowercase
    - replace greek rho with "rho"
    - remove non-alphanumerics
    """
    if not text:
        return ""
    t = text.lower()
    t = t.replace("ρ", "rho")
    # collapse common separators to nothing
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _load_eval_param_keys(cache_path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not cache_path.exists():
        return keys
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []) or []:
            params = row.get("params") or {}
            if isinstance(params, dict):
                for k in params.keys():
                    if isinstance(k, str):
                        keys.add(k)
    except Exception:
        return keys
    return keys


def default_registry(persist_dir: str = "./data/vectorstore") -> List[PropertyEntry]:
    """
    Build registry from what is available. We still define canonical properties,
    but we only enable the ones backed by caches/tools present in the local store.
    """
    p = Path(persist_dir)
    eval_cache = p / "eval_params_cache.json"
    petro_cache = p / "petro_params_cache.json"
    facts_cache = p / "facts_cache.json"

    eval_keys = _load_eval_param_keys(eval_cache)
    has_eval = eval_cache.exists() and len(eval_keys) > 0
    has_petro = petro_cache.exists()
    has_facts = facts_cache.exists()

    entries: List[PropertyEntry] = []

    if has_eval:
        # Map the stable eval-params keys into canonical properties.
        # These are *not* synonyms; they are canonical concepts for deterministic lookup.
        if "Rhoma" in eval_keys:
            entries.append(PropertyEntry("matrix_density", "lookup_evaluation_parameters", structured_key="Rhoma", display="Matrix density (ρma / Rhoma)"))
        if "Rhofl" in eval_keys:
            entries.append(PropertyEntry("fluid_density", "lookup_evaluation_parameters", structured_key="Rhofl", display="Fluid density (ρfl / Rhofl)"))
        if "Grmin" in eval_keys:
            entries.append(PropertyEntry("grmin", "lookup_evaluation_parameters", structured_key="Grmin", display="GRmin"))
        if "Grmax" in eval_keys:
            entries.append(PropertyEntry("grmax", "lookup_evaluation_parameters", structured_key="Grmax", display="GRmax"))
        if "a" in eval_keys:
            entries.append(PropertyEntry("archie_a", "lookup_evaluation_parameters", structured_key="a", display="Archie a (tortuosity)"))
        if "m" in eval_keys:
            entries.append(PropertyEntry("archie_m", "lookup_evaluation_parameters", structured_key="m", display="Archie m (cementation exponent)"))
        if "n" in eval_keys:
            entries.append(PropertyEntry("archie_n", "lookup_evaluation_parameters", structured_key="n", display="Archie n (saturation exponent)"))
        if "A" in eval_keys:
            entries.append(PropertyEntry("A", "lookup_evaluation_parameters", structured_key="A", display="A (evaluation constant)"))
        if "B" in eval_keys:
            entries.append(PropertyEntry("B", "lookup_evaluation_parameters", structured_key="B", display="B (evaluation constant)"))

    if has_petro:
        entries.extend(
            [
                PropertyEntry("netgros", "lookup_petrophysical_params", structured_key="netgros", display="Net/Gross (NetGros)"),
                PropertyEntry("phif", "lookup_petrophysical_params", structured_key="phif", display="PHIF (porosity)"),
                PropertyEntry("sw", "lookup_petrophysical_params", structured_key="sw", display="SW (water saturation)"),
                PropertyEntry("klogh", "lookup_petrophysical_params", structured_key="klogh", display="KLOGH (permeability)"),
            ]
        )

    if has_facts:
        # Facts are generic; the resolver routes to lookup_structured_facts.
        entries.extend(
            [
                PropertyEntry("rw", "lookup_structured_facts", structured_key=None, display="Rw"),
                PropertyEntry("temperature_gradient", "lookup_structured_facts", structured_key=None, display="Temperature gradient"),
                PropertyEntry("reservoir_temperature", "lookup_structured_facts", structured_key=None, display="Reservoir temperature"),
                PropertyEntry("cutoff", "lookup_structured_facts", structured_key=None, display="Cutoff"),
            ]
        )

    return entries


def _rho_shorthand_key(text: str) -> Optional[str]:
    """
    Deterministically interpret rho-shorthand without requiring synonyms:
    - "rho ma" / "rho_ma" / "ρma" -> "rhoma"
    - "rho fl" / "rho_fl" / "ρfl" -> "rhofl"
    """
    t = canonical_tokenize(text)
    if not t:
        return None
    # direct rhoma / rhofl
    if "rhoma" in t:
        return "rhoma"
    if "rhofl" in t:
        return "rhofl"
    # split forms: rho + ma/fl
    if "rho" in t and "ma" in t:
        return "rhoma"
    if "rho" in t and "fl" in t:
        return "rhofl"
    return None


def resolve_property_deterministic(question: str, registry: Sequence[PropertyEntry]) -> Tuple[Optional[PropertyEntry], List[PropertyEntry]]:
    """
    Return (best_match, candidates). If best_match is None, candidates may be used for clarification.
    """
    q = question or ""
    ql = q.lower()
    qt = canonical_tokenize(q)

    # 1) Deterministic rho shorthand
    rho_key = _rho_shorthand_key(q)
    if rho_key:
        for e in registry:
            if e.structured_key and canonical_tokenize(e.structured_key) == rho_key:
                return e, []

    # 2) Direct mention of structured keys (e.g., "grmax", "rhoma", "phif")
    # First, check if single-letter matches are part of a well name (e.g., "15/9-F-15 A")
    # to avoid misinterpreting well suffixes as Archie parameters
    well_pattern = re.search(r"15[\s_/-]*9[\s_/-]*(?:F[\s_/-]*-?[\s_/-]*)?-?\s*\d+\s*([A-Z])\b", question or "", re.IGNORECASE)
    well_suffix = well_pattern.group(1).upper() if well_pattern else None
    
    for e in registry:
        if e.structured_key:
            sk = canonical_tokenize(e.structured_key)
            if not sk:
                continue
            # Avoid pathological matches for single-letter keys (A/B/a/m/n) inside arbitrary words.
            if len(sk) == 1:
                # CRITICAL: Don't match single letters that are well name suffixes
                if well_suffix and e.structured_key.upper() == well_suffix:
                    continue  # Skip - this is a well suffix, not a property
                # Require whole-word match in the raw question.
                if re.search(rf"\b{re.escape(e.structured_key)}\b", question or "", flags=re.IGNORECASE):
                    return e, []
                continue
            if sk in qt:
                return e, []

    # 3) Deterministic canonical phrases (small, stable set)
    # These are not endless synonyms; they are controlled "concept labels" we support.
    phrase_map = {
        "matrixdensity": "matrix_density",
        "fluiddensity": "fluid_density",
        "nettogross": "netgros",
        "netgross": "netgros",
        "watersaturation": "sw",
        "temperaturegradient": "temperature_gradient",
        "reservoirtemperature": "reservoir_temperature",
        "permeability": "klogh",
        "porosity": "phif",
        "perm": "klogh",
        "poro": "phif",
    }
    for needle, canon in phrase_map.items():
        if needle in qt:
            for e in registry:
                if e.canonical == canon:
                    return e, []

    # 3b) Token-level fuzzy intent for common domain terms (typo tolerant, bounded).
    # This avoids hardcoding specific typos while still resolving things like:
    # - "mtrx dnetsiy" -> matrix density
    try:
        from rapidfuzz import fuzz  # type: ignore

        tokens = re.findall(r"[a-z]+", ql)
        if tokens:
            den = max(fuzz.ratio(t, "density") for t in tokens)
            mat = max(fuzz.ratio(t, "matrix") for t in tokens)
            flu = max(fuzz.ratio(t, "fluid") for t in tokens)

            if den >= 70 and mat >= 75:
                e = next((x for x in registry if x.canonical == "matrix_density"), None)
                if e:
                    return e, []
            if den >= 70 and flu >= 75:
                e = next((x for x in registry if x.canonical == "fluid_density"), None)
                if e:
                    return e, []
    except Exception:
        pass

    # 4) Candidate shortlist for ambiguity handling
    candidates: List[PropertyEntry] = []
    # If query contains "density" and "rho", suggest both densities
    # also allow "dens*" to catch common typos like "densty" or "densiy"
    if ("density" in ql) or ("rho" in qt) or re.search(r"\bdens", ql):
        candidates.extend([e for e in registry if e.canonical in {"matrix_density", "fluid_density"}])
    # If query contains "archie", suggest archie params
    if "archie" in ql:
        candidates.extend([e for e in registry if e.canonical in {"archie_a", "archie_m", "archie_n"}])
    # If query contains "gr", suggest GRmin/GRmax
    if "gr" in ql:
        candidates.extend([e for e in registry if e.canonical in {"grmin", "grmax"}])
    # If query contains petro tokens, suggest petro params
    if any(k in ql for k in ["phif", "phi", "poro", "net", "gross", "sw", "klogh", "permeab", "perm"]) or re.search(r'\bk\b', ql, re.IGNORECASE):
        candidates.extend([e for e in registry if e.tool == "lookup_petrophysical_params"])

    # Dedup while preserving order
    seen: Set[str] = set()
    uniq: List[PropertyEntry] = []
    for e in candidates:
        if e.canonical in seen:
            continue
        seen.add(e.canonical)
        uniq.append(e)

    # 5) Fuzzy match over a bounded surface-form registry (typo tolerant).
    # This is the preferred "smart" path: algorithmic typo handling, not manual substitutions.
    try:
        from rapidfuzz import fuzz, process  # type: ignore

        # Build surface forms for each property entry.
        # We keep this bounded to prevent hallucinations.
        surface: List[Tuple[str, PropertyEntry]] = []

        def add_surface(s: str, entry: PropertyEntry):
            ss = (s or "").strip()
            if not ss:
                return
            surface.append((ss, entry))

        for e in registry:
            add_surface(e.canonical.replace("_", " "), e)
            add_surface(e.canonical, e)
            if e.structured_key:
                add_surface(e.structured_key, e)
            if e.display:
                add_surface(e.display, e)
            # Stable concept labels (not endless synonyms)
            if e.canonical == "matrix_density":
                add_surface("matrix density", e)
                add_surface("rho ma", e)
                add_surface("ρma", e)
            if e.canonical == "fluid_density":
                add_surface("fluid density", e)
                add_surface("rho fl", e)
                add_surface("ρfl", e)
            if e.canonical == "netgros":
                add_surface("net to gross", e)
            if e.canonical == "temperature_gradient":
                add_surface("temperature gradient", e)
            if e.canonical == "klogh":
                add_surface("perm", e)
                add_surface("permeability", e)
                add_surface("k", e)
            if e.canonical == "phif":
                add_surface("poro", e)
                add_surface("phi", e)
                add_surface("porosity", e)

        choices = [s for s, _ in surface]
        q = (question or "").strip()
        if q and choices:
            # Robust fuzzy scoring: combine multiple scorers and then aggregate to distinct entries.
            scorers = [fuzz.WRatio]
            if len(q) >= 20:
                # For longer queries, also consider substring alignment.
                scorers.append(fuzz.partial_ratio)

            entry_best: Dict[str, Tuple[float, PropertyEntry]] = {}
            for sc in scorers:
                hits = process.extract(q, choices, scorer=sc, limit=10)
                for _, score, idx in hits:
                    entry = surface[idx][1]
                    prev = entry_best.get(entry.canonical)
                    if (prev is None) or (score > prev[0]):
                        entry_best[entry.canonical] = (float(score), entry)

            ranked = sorted(entry_best.values(), key=lambda t: t[0], reverse=True)
            if ranked:
                top_score, top_entry = ranked[0]
                second_score = ranked[1][0] if len(ranked) > 1 else 0.0

                # Strong accept: high score + clear margin.
                STRONG_THRESH = 86.0
                STRONG_MARGIN = 6.0
                # Weak accept: moderate score but very clear separation (handles heavy typos like "dnetsiy").
                WEAK_THRESH = 74.0
                WEAK_MARGIN = 18.0

                if top_score >= STRONG_THRESH and (top_score - second_score) >= STRONG_MARGIN:
                    return top_entry, []
                if top_score >= WEAK_THRESH and (top_score - second_score) >= WEAK_MARGIN:
                    return top_entry, []

                # Otherwise return the top candidates for clarification (bounded).
                cand_entries: List[PropertyEntry] = []
                for score, entry in ranked:
                    if score < 70.0:
                        continue
                    cand_entries.append(entry)
                if cand_entries:
                    return None, cand_entries
    except Exception:
        # RapidFuzz not installed or failed; fall back to heuristic candidates
        pass

    return None, uniq


