"""
Bounded entity resolver (well + formation + property).

Smart pattern:
- Deterministic extraction + fuzzy candidate generation from *known registries* (no guessing).
- If unresolved/ambiguous, use an LLM ONLY to choose among candidates or ask a clarification.
- Downstream tools remain deterministic (no numeric hallucinations).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_openai import ChatOpenAI

from .property_registry import PropertyEntry, default_registry, resolve_property_deterministic
from .query_normalizer import extract_well, normalize_formation


@dataclass(frozen=True)
class ScoredCandidate:
    value: str
    score: float


@dataclass(frozen=True)
class EntityResolution:
    well: Optional[str]
    formation: Optional[str]
    property: Optional[str]  # canonical property key
    tool: Optional[str]
    needs_clarification: bool
    clarification_question: Optional[str]


def _load_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _norm_key(s: str) -> str:
    return re.sub(r"[^0-9A-Z]+", "", (s or "").upper())


@lru_cache(maxsize=1)
def _build_registries(persist_dir: str = "./data/vectorstore") -> Tuple[List[str], Dict[str, List[str]], List[str], List[PropertyEntry]]:
    """
    Returns:
    - wells: list of canonical well ids (e.g., 15/9-F-5)
    - well_to_formations: well -> formations list (strings as stored in caches)
    - formations: global unique formation list
    - properties: PropertyEntry list (from property registry)
    """
    persist = Path(persist_dir)

    wells: Dict[str, None] = {}
    well_to_forms: Dict[str, Dict[str, None]] = {}
    forms: Dict[str, None] = {}

    # 1) Well picks cache (global, not under persist_dir)
    wp = _load_json(Path("./data/well_picks_cache.json"))
    if wp and isinstance(wp.get("rows"), list):
        for r in wp["rows"]:
            if not isinstance(r, dict):
                continue
            w_raw = r.get("well") or ""
            f_raw = r.get("formation") or ""
            w = extract_well(str(w_raw))
            if w:
                wells[w] = None
                well_to_forms.setdefault(w, {})
                if isinstance(f_raw, str) and f_raw.strip():
                    well_to_forms[w][f_raw.strip()] = None
                    forms[f_raw.strip()] = None

    # 2) Eval params cache (persist_dir)
    ev = _load_json(persist / "eval_params_cache.json")
    if ev and isinstance(ev.get("rows"), list):
        for r in ev["rows"]:
            if not isinstance(r, dict):
                continue
            w = r.get("well")
            if isinstance(w, str) and w.strip():
                wells[w.strip()] = None
                well_to_forms.setdefault(w.strip(), {})
            fl = r.get("formations") or []
            if isinstance(fl, list) and isinstance(w, str) and w.strip():
                for f in fl:
                    if isinstance(f, str) and f.strip():
                        well_to_forms[w.strip()][f.strip()] = None
                        forms[f.strip()] = None

    # 3) Petro params cache (persist_dir)
    pp = _load_json(persist / "petro_params_cache.json")
    if pp and isinstance(pp.get("rows"), list):
        for r in pp["rows"]:
            if not isinstance(r, dict):
                continue
            w = r.get("well")
            f = r.get("formation")
            if isinstance(w, str) and w.strip():
                wells[w.strip()] = None
                well_to_forms.setdefault(w.strip(), {})
                if isinstance(f, str) and f.strip():
                    well_to_forms[w.strip()][f.strip()] = None
                    forms[f.strip()] = None

    # Materialize
    wells_list = sorted(wells.keys())
    well_to_forms_list: Dict[str, List[str]] = {w: sorted(d.keys()) for w, d in well_to_forms.items()}
    forms_list = sorted(forms.keys())
    props = default_registry(persist_dir)
    return wells_list, well_to_forms_list, forms_list, props


def _topk_well_candidates(query: str, wells: Sequence[str], k: int = 8) -> List[ScoredCandidate]:
    """
    Typos in wells are usually separator/format issues; match on normalized alnum key.
    """
    try:
        from rapidfuzz import fuzz  # type: ignore
    except Exception:
        return []

    qk = _norm_key(query)
    scored: List[ScoredCandidate] = []
    for w in wells:
        wk = _norm_key(w)
        if not wk:
            continue
        s = max(float(fuzz.partial_ratio(qk, wk)), float(fuzz.ratio(qk, wk)))
        scored.append(ScoredCandidate(value=w, score=s))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:k]


def _topk_text_candidates(query: str, choices: Sequence[str], k: int = 8) -> List[ScoredCandidate]:
    try:
        from rapidfuzz import fuzz, process  # type: ignore
    except Exception:
        return []

    if not query:
        return []
    # case-insensitive matching
    lower_map = {c.lower(): c for c in choices if isinstance(c, str) and c.strip()}
    if not lower_map:
        return []
    hits = process.extract(query.lower(), list(lower_map.keys()), scorer=fuzz.partial_ratio, limit=k)
    out: List[ScoredCandidate] = []
    for choice_l, score, _ in hits:
        out.append(ScoredCandidate(value=lower_map.get(choice_l, choice_l), score=float(score)))
    return out


def _accept_or_ambiguous(cands: Sequence[ScoredCandidate], thresh: float, margin: float) -> Tuple[Optional[str], List[str]]:
    if not cands:
        return None, []
    top = cands[0]
    second = cands[1].score if len(cands) > 1 else 0.0
    if top.score >= thresh and (top.score - second) >= margin:
        return top.value, []
    # ambiguous: return the top few values
    return None, [c.value for c in cands[:5]]


def _agent_enabled() -> bool:
    return os.getenv("RAG_ENTITY_RESOLVER", "true").lower() in {"1", "true", "yes"}


def resolve_with_bounded_agent(
    question: str,
    persist_dir: str = "./data/vectorstore",
) -> EntityResolution:
    """
    Resolve well + formation + property from a possibly-typoed question.
    """
    q = question or ""
    wells, well_to_forms, forms, props = _build_registries(persist_dir)

    # Deterministic parse first
    well = extract_well(q)
    formation = normalize_formation(q)
    prop_entry, prop_candidates = resolve_property_deterministic(q, props)

    prop = prop_entry.canonical if prop_entry else None
    tool = prop_entry.tool if prop_entry else None

    # Candidate generation only if needed
    well_amb: List[str] = []
    if not well:
        w_cands = _topk_well_candidates(q, wells, k=8)
        well, well_amb = _accept_or_ambiguous(w_cands, thresh=90.0, margin=8.0)

    formation_amb: List[str] = []
    if not formation:
        # If we have a well, restrict formation candidates to that well
        choices = well_to_forms.get(well, []) if well else forms
        f_cands = _topk_text_candidates(q, choices, k=8)
        formation, formation_amb = _accept_or_ambiguous(f_cands, thresh=84.0, margin=8.0)

    property_amb: List[str] = []
    if prop is None and prop_candidates:
        property_amb = [c.canonical for c in prop_candidates][:8]

    # If we have property but no well, we must ask (can’t answer without a well)
    if prop and not well:
        if well_amb:
            return EntityResolution(
                well=None,
                formation=formation,
                property=prop,
                tool=tool,
                needs_clarification=True,
                clarification_question=f"I couldn’t confidently identify the well. Did you mean: {', '.join(well_amb[:5])}?",
            )
        return EntityResolution(
            well=None,
            formation=formation,
            property=prop,
            tool=tool,
            needs_clarification=True,
            clarification_question="Which well did you mean (e.g., 15/9-F-4)?",
        )

    # If everything resolved deterministically
    if well and (prop or prop_entry) and (formation or True):
        return EntityResolution(well=well, formation=formation, property=prop, tool=tool, needs_clarification=False, clarification_question=None)

    # If ambiguous/missing and agent enabled, ask the agent to choose among bounded candidates or ask a clarification.
    if not _agent_enabled():
        # If no agent, ask a deterministic question
        if not well:
            return EntityResolution(None, formation, prop, tool, True, "Which well did you mean (e.g., 15/9-F-4)?")
        if prop is None:
            return EntityResolution(well, formation, None, None, True, "Which parameter/property do you mean (e.g., matrix density or fluid density)?")
        return EntityResolution(well, formation, prop, tool, True, "Can you clarify the formation/property?")

    # Build bounded candidate lists
    well_opts = [well] if well else (well_amb or [])
    if not well_opts:
        well_opts = wells[:10]
    if well and well in well_to_forms:
        form_opts = list(well_to_forms.get(well, []))[:20]
    else:
        form_opts = formation_amb or forms[:20]

    prop_opts: List[dict] = []
    if prop_entry:
        prop_opts = [{"canonical": prop_entry.canonical, "label": prop_entry.display or prop_entry.canonical, "tool": prop_entry.tool}]
    else:
        # Use either the candidates from resolver, or the whole registry (bounded)
        if prop_candidates:
            use = prop_candidates
        else:
            use = props[:12]
        for e in use:
            prop_opts.append({"canonical": e.canonical, "label": e.display or e.canonical, "tool": e.tool})

    model = os.getenv("RAG_ENTITY_RESOLVER_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = (
        "You are an entity resolver for a deterministic petrophysical QA system.\n"
        "Your job is to map the user's text to canonical entities.\n\n"
        "Rules:\n"
        "- You MUST choose only from the provided candidates.\n"
        "- If you cannot uniquely choose, ask ONE clarification question.\n"
        "- Do NOT answer with numeric values.\n\n"
        f"User text: {q}\n\n"
        f"Well candidates: {well_opts}\n"
        f"Formation candidates: {form_opts}\n"
        f"Property candidates: {prop_opts}\n\n"
        "Return JSON with keys:\n"
        "{well: string|null, formation: string|null, property: string|null, tool: string|null,\n"
        " needs_clarification: boolean, clarification_question: string|null}\n"
    )

    resp = llm.invoke([{"role": "user", "content": prompt}])
    txt = (resp.content or "").strip()

    # Parse JSON robustly
    m = re.search(r"\{[\s\S]*\}", txt)
    blob = m.group(0) if m else txt
    try:
        data = json.loads(blob)
    except Exception:
        # If parsing fails, ask a deterministic clarification
        return EntityResolution(well, formation, prop, tool, True, "Can you clarify the well, formation, and parameter?")

    needs = bool(data.get("needs_clarification"))
    clar = data.get("clarification_question")
    rw = data.get("well")
    rf = data.get("formation")
    rp = data.get("property")
    rt = data.get("tool")

    # Validate boundedness
    if rw is not None and rw not in well_opts:
        rw = None
        needs = True
    if rf is not None and rf not in form_opts:
        rf = None
        needs = True
    if rp is not None and not any(o.get("canonical") == rp for o in prop_opts):
        rp = None
        needs = True
    if rt is not None and not any(o.get("canonical") == rp and o.get("tool") == rt for o in prop_opts):
        rt = None

    if needs and not clar:
        clar = "Can you clarify the well, formation, and parameter?"

    return EntityResolution(
        well=rw or well,
        formation=rf or formation,
        property=rp or prop,
        tool=rt or tool,
        needs_clarification=needs,
        clarification_question=clar,
    )


