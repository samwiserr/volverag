"""
Optional agentic disambiguation for property mapping.

This is used ONLY when deterministic normalization can't map a property, and we need
to choose between a small set of candidates. The agent never produces numeric answers.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ..core.model_factory import get_chat_model
from .property_registry import PropertyEntry


@dataclass(frozen=True)
class Disambiguation:
    canonical: Optional[str]
    confidence: float
    clarification_question: Optional[str]


def _enabled() -> bool:
    return os.getenv("RAG_AGENT_DISAMBIGUATE", "true").lower() in {"1", "true", "yes"}


def choose_property_with_agent(question: str, candidates: List[PropertyEntry]) -> Disambiguation:
    if not _enabled() or not candidates:
        return Disambiguation(canonical=None, confidence=0.0, clarification_question=None)

    model = os.getenv("RAG_AGENT_DISAMBIGUATE_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    llm = get_chat_model(model, temperature=0, role="disambiguation")

    options = "\n".join([f"- {c.canonical}: {c.display or c.canonical}" for c in candidates])
    prompt = (
        "You are a routing assistant for a deterministic petrophysical QA system.\n"
        "Task: choose the best matching canonical property from the options below.\n"
        "Rules:\n"
        "- ONLY choose from the listed options.\n"
        "- If the user's wording is not specific enough to select ONE option, respond with canonical=null and propose a short clarification question.\n"
        "- Prefer asking ONE clarification question over guessing.\n"
        "- Example: if the user asks for 'density' and the options include both matrix density and fluid density, you MUST ask which one.\n"
        "- Do NOT answer the user question with values.\n\n"
        f"User question: {question}\n\n"
        f"Options:\n{options}\n\n"
        "Respond in JSON with keys: canonical (string or null), confidence (0..1), clarification_question (string or null)."
    )

    resp = llm.invoke([{"role": "user", "content": prompt}])
    txt = (resp.content or "").strip()
    try:
        # The model may wrap JSON in prose; extract the first JSON object.
        m = re.search(r"\{[\s\S]*\}", txt)
        blob = m.group(0) if m else txt
        data = json.loads(blob)
        canonical = data.get("canonical")
        conf = float(data.get("confidence", 0.0) or 0.0)
        clar = data.get("clarification_question")
        if canonical is not None and not any(c.canonical == canonical for c in candidates):
            canonical = None
            conf = 0.0
        return Disambiguation(canonical=canonical, confidence=conf, clarification_question=clar)
    except Exception:
        return Disambiguation(canonical=None, confidence=0.0, clarification_question=None)


