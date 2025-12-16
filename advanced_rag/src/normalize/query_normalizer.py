"""
Normalize -> Resolve helpers.

This module converts a raw user question into a canonical internal representation:
- well identifier normalized (15/9-F-4 style)
- formation normalized to known vocabulary (best-effort)
- property normalized via synonym resolution (e.g., "density of fluid" -> "fluid_density")
- intent classification for deterministic routing

The goal is to remove linguistic variability BEFORE retrieval/tool selection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


@dataclass(frozen=True)
class NormalizedQuery:
    raw: str
    well: Optional[str]
    formation: Optional[str]
    property: Optional[str]  # canonical property key (registry canonical)
    tool: Optional[str]  # preferred structured tool name
    intent: str  # "fact" | "list" | "section" | "unknown"


def _canonicalize_well(s: str) -> str:
    """Normalize well string for matching across formats. Supports any well format, not just 15/9."""
    w = s.strip().upper()
    w = w.replace("_", "/").replace(" ", "")
    # Normalize any XX-YY to XX/YY pattern
    w = re.sub(r"(\d+)-(\d+)", r"\1/\2", w)
    # Normalize XX/YYF to XX/YY-F (platform wells)
    w = re.sub(r"(\d+/\d+)([A-Z])(\d)", r"\1-\2-\3", w)
    # Ensure dash between letter and digits when missing: XX/YY-F5 -> XX/YY-F-5
    w = re.sub(r"(\d+/\d+-[A-Z])(\d)", r"\1-\2", w)
    # Handle common 15/9 patterns (backward compatibility)
    w = w.replace("15-9", "15/9")
    w = w.replace("15/9F", "15/9-F")
    w = w.replace("--", "-")
    return w


def extract_well(text: str) -> Optional[str]:
    """
    Extract well name from query. Supports multiple well formats:
    - Platform wells: 15/9-F-5, 15/9-F-15 A
    - Non-platform wells: 15/9-19A, 19/9-19 bt2
    - With "Well NO" prefix: Well NO 15/9-F-5
    """
    # Generic pattern: match any well format like "XX/YY-ZZZ" or "XX/YY-ABC-ZZZ"
    # Pattern 1: Platform format (XX/YY-F-NN or XX/YY-F-NN A)
    m = re.search(r"(?:Well\s+NO\s+)?(\d+[\s_/-]*\d+[\s_/-]*[A-Z][\s_/-]*-?\s*\d+(?:\s+[A-Z0-9]+|[A-Z0-9]+)?)\b", text, re.IGNORECASE)
    if m:
        return _canonicalize_well(m.group(1))
    
    # Pattern 2: Non-platform format (XX/YY-NNN or XX/YY-NNN suffix)
    m2 = re.search(r"(?:Well\s+NO\s+)?(\d+[\s_/-]*\d+[\s_/-]*-?\s*\d+(?:\s+[A-Z0-9]+|[A-Z0-9]+)?)\b", text, re.IGNORECASE)
    if m2:
        return _canonicalize_well(m2.group(1))
    
    return None


@lru_cache(maxsize=1)
def _formation_vocab() -> Set[str]:
    """
    Build a controlled vocabulary from available caches so we can normalize formations.
    """
    vocab: Set[str] = set()

    # From petro params cache (values are base formation names)
    petro = Path("./data/vectorstore/petro_params_cache.json")
    if petro.exists():
        try:
            payload = json.loads(petro.read_text(encoding="utf-8"))
            for r in payload.get("rows", []):
                f = r.get("formation")
                if isinstance(f, str) and f.strip():
                    vocab.add(f.strip())
        except Exception:
            pass

    # From well picks cache (formation names include "Fm.", groups, etc.)
    picks = Path("./data/well_picks_cache.json")
    if picks.exists():
        try:
            payload = json.loads(picks.read_text(encoding="utf-8"))
            for r in payload.get("rows", []):
                f = r.get("formation")
                if isinstance(f, str) and f.strip():
                    vocab.add(f.strip())
        except Exception:
            pass

    # From eval params cache (base formation names)
    evalc = Path("./data/vectorstore/eval_params_cache.json")
    if evalc.exists():
        try:
            payload = json.loads(evalc.read_text(encoding="utf-8"))
            for r in payload.get("rows", []):
                for f in r.get("formations", []) or []:
                    if isinstance(f, str) and f.strip():
                        vocab.add(f.strip())
        except Exception:
            pass

    return {v for v in vocab if len(v) >= 3}


def normalize_formation(text: str) -> Optional[str]:
    """
    Normalize formation to the closest known token.
    - Prefer exact substring match (fast, deterministic)
    - Fall back to fuzzy match over a bounded vocabulary (typo tolerant)
    """
    ql = text.lower()
    # quick heuristics for common "Formation"/"Fm" phrasing
    ql = ql.replace("formation", " ").replace("fm.", " ").replace("fm", " ")
    vocab = _formation_vocab()
    # Longest match wins (avoids matching "Ty" inside other words)
    best = None
    for f in sorted(vocab, key=len, reverse=True):
        if f.lower() in ql:
            best = f
            break
    if best:
        # Prefer base names for some cases (e.g., "Sleipner Fm." -> "Sleipner")
        return best

    # Fuzzy fallback (bounded to vocab): handles typos like "hugni" -> "Hugin"
    try:
        from rapidfuzz import fuzz, process  # type: ignore

        if vocab:
            # Use partial_ratio because the query contains many other words.
            # IMPORTANT: compare case-insensitively (vocab entries are Title Case).
            lower_map = {}
            for v in vocab:
                if isinstance(v, str) and v.strip():
                    lower_map[v.lower()] = v
            choices = list(lower_map.keys())

            hit = process.extract(ql, choices, scorer=fuzz.partial_ratio, limit=5)
            if hit:
                best_choice_l, best_score, _ = hit[0]
                second_score = hit[1][1] if len(hit) > 1 else 0.0
                # Slightly lower threshold with a stronger margin: this is bounded to a known
                # formation vocabulary so we can be typo-tolerant without guessing.
                THRESH = 84.0
                MARGIN = 8.0
                if best_score >= THRESH and (best_score - second_score) >= MARGIN:
                    return lower_map.get(best_choice_l, best_choice_l)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _property_synonyms() -> Dict[str, str]:
    """
    Map many user phrases to canonical property keys.
    Canonical keys are used to deterministically route to tools.
    """
    return {
        # Evaluation parameters
        "matrix density": "matrix_density",
        "density of matrix": "matrix_density",
        "rhoma": "matrix_density",
        "ρma": "matrix_density",
        "fluid density": "fluid_density",
        "density of fluid": "fluid_density",
        "density fluid": "fluid_density",
        "rho fluid": "fluid_density",
        "rhofl": "fluid_density",
        "ρfl": "fluid_density",
        "grmax": "grmax",
        "gr min": "grmin",
        "grmin": "grmin",
        "archie a": "archie_a",
        "archie m": "archie_m",
        "archie n": "archie_n",
        "tortuosity factor": "archie_a",
        "cementation exponent": "archie_m",
        "saturation exponent": "archie_n",
        # Petro params table
        "net to gross": "netgros",
        "net-to-gross": "netgros",
        "net/gross": "netgros",
        "ntg": "netgros",
        "phif": "phif",
        "porosity": "phif",
        "phi": "phif",
        "water saturation": "sw",
        "sw": "sw",
        "klogh": "klogh",
        "permeability": "klogh",
        "perm": "klogh",
        "poro": "phif",
        # Facts
        "rw": "rw",
        "temperature gradient": "temperature_gradient",
        "reservoir temperature": "reservoir_temperature",
        "cutoff": "cutoff",
        "cut-off": "cutoff",
        "cut off": "cutoff",
    }


def normalize_property(text: str) -> Optional[str]:
    ql = text.lower()
    syn = _property_synonyms()
    # Prefer longest phrase match
    for phrase in sorted(syn.keys(), key=len, reverse=True):
        if phrase in ql:
            return syn[phrase]
    return None


def normalize_query(text: str) -> NormalizedQuery:
    q = text or ""
    ql = q.lower()
    well = extract_well(q)
    formation = normalize_formation(q)
    prop = normalize_property(q)

    # Simple intent classification
    intent = "unknown"
    if any(k in ql for k in ["summary", "introduction", "conclusion", "abstract"]) and well:
        intent = "section"
    elif prop is not None and well is not None:
        intent = "fact"
    elif any(k in ql for k in ["list", "formations", "formation"]) and well:
        intent = "list"

    # Resolve preferred tool using the data-driven registry
    tool = None
    try:
        from .property_registry import default_registry, resolve_property_deterministic
        from .agent_disambiguator import choose_property_with_agent

        registry = default_registry("./data/vectorstore")
        entry, candidates = resolve_property_deterministic(q, registry)
        if entry is None and candidates:
            # Agent used only for ambiguity; it never outputs values.
            dis = choose_property_with_agent(q, candidates)
            if dis.canonical:
                entry = next((e for e in registry if e.canonical == dis.canonical), None)
        if entry:
            prop = entry.canonical
            tool = entry.tool
            intent = "fact" if well else intent
    except Exception:
        # Keep best-effort behavior if normalization registry isn't available
        pass

    return NormalizedQuery(raw=q, well=well, formation=formation, property=prop, tool=tool, intent=intent)


