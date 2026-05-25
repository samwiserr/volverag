"""
HyDE — Hypothetical Document Embeddings (Gao et al., SIGIR 2023).

Instead of embedding the raw user query (which is typically short and
vocabulary-poor), HyDE asks an LLM to write a *hypothetical document* that
would perfectly answer the query.  That synthetic document is then embedded
and used for retrieval.  Because the hypothetical answer is phrased the way
domain reports actually are written, it lands much closer to the real passage
in embedding space.

Usage inside RetrieverTool:
    expander = HyDEExpander()
    hyde_text = expander.expand(query)   # returns the synthetic document
    # embed hyde_text instead of query for dense retrieval

Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance
           Labels" (SIGIR 2023). https://arxiv.org/abs/2212.10496
"""
from __future__ import annotations

import logging
from typing import Optional

from ..core.model_factory import get_chat_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_HYDE_PROMPT = """\
You are an expert petrophysicist writing a technical report passage for the
Equinor Volve oilfield (North Sea, Norway).

A user asked: "{query}"

Write a concise technical passage (3–5 sentences) that would *directly* and
*accurately* answer this question in the style of a Volve petrophysical
report.  Include likely values, formation names, or well identifiers if
plausible.  Do NOT include disclaimers.  Write as if this passage came from
the actual report."""


class HyDEExpander:
    """
    Generates a hypothetical answer document for a user query and returns
    both the generated text and (optionally) an averaged embedding that
    combines the query and hypothetical answer for better retrieval coverage.

    Design choices:
    - Uses a fast model (Groq instant by default) to keep latency low.
    - Can return multiple hypothetical docs (``n_hypotheses``) and average
      their embeddings — similar to the original HyDE paper.
    - Falls back gracefully: if generation fails, the original query is used.
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        n_hypotheses: int = 1,
        temperature: float = 0.4,
    ) -> None:
        """
        Args:
            model: Chat model used for hypothesis generation.
            n_hypotheses: Number of independent hypothetical documents to
                generate.  Their embeddings are averaged for retrieval.
            temperature: Slight diversity helps cover paraphrase space.
        """
        self._llm = get_chat_model(
            model,
            temperature=temperature,
            max_tokens=300,
            role="hyde",
        )
        self._n = n_hypotheses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> str:
        """
        Return a single hypothetical document string.

        If ``n_hypotheses > 1``, returns the first generated document;
        use ``expand_all`` to get all of them.
        """
        docs = self.expand_all(query)
        return docs[0] if docs else query

    def expand_all(self, query: str) -> list[str]:
        """
        Return *n_hypotheses* hypothetical documents for ``query``.

        Falls back to ``[query]`` on any error so the caller never crashes.
        """
        results: list[str] = []
        prompt = _HYDE_PROMPT.format(query=query)
        for i in range(self._n):
            try:
                response = self._llm.invoke([{"role": "user", "content": prompt}])
                text = response.content.strip()
                if text:
                    results.append(text)
                    logger.debug(f"[HyDE] Hypothesis {i+1}: {text[:120]}…")
            except Exception as exc:
                logger.warning(f"[HyDE] Generation failed on attempt {i+1}: {exc}")
        return results if results else [query]

    def build_retrieval_queries(self, query: str) -> list[str]:
        """
        Return a merged list of retrieval queries:
            [original_query] + [hypothetical_doc_1, …]

        This is passed to ``_hybrid_retrieve`` for multi-query fusion, so
        both keyword precision (BM25 on the raw query) and semantic recall
        (dense on the hypothetical doc) are maximised.
        """
        hypotheses = self.expand_all(query)
        # Always keep the raw query first so BM25 benefits from exact terms
        queries = [query] + hypotheses
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            k = q.strip().lower()
            if k not in seen:
                seen.add(k)
                deduped.append(q.strip())
        return deduped
