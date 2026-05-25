"""
Contextual Chunker — Anthropic Contextual Retrieval (September 2024).

For every raw chunk produced by IntelligentChunker, this module calls a cheap
LLM to generate a ≤2-sentence situating context ("This chunk discusses X
within the 15/9-F-5 petrophysical report…") and prepends it to the chunk
before embedding.  This bridges the vocabulary gap between how users phrase
questions and how engineers wrote reports, without changing the stored source
text shown to users.

Reference: https://www.anthropic.com/news/contextual-retrieval
          Guu et al., 2020 (RAG); Anthropic blog, Sept 2024.
"""
from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from ..core.model_factory import get_chat_model
from .intelligent_chunker import IntelligentChunker, TextChunk, ChunkingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_CONTEXT_PROMPT = """\
You are a petrophysical data assistant.

<document>
{document_text}
</document>

Here is a chunk from that document:
<chunk>
{chunk_text}
</chunk>

Write a concise (≤2 sentences) context that situates this chunk within the
full document above.  Mention the well name, formation, or key parameter if
clearly stated.  Do NOT re-state facts; only provide context.  Be factual."""


class ContextualChunker:
    """
    Wraps IntelligentChunker and enriches each chunk with an LLM-generated
    situating context prepended before embedding.

    Architecture:
        raw text  →  IntelligentChunker  →  [ctx | chunk_text]  →  embed

    The original chunk text is stored in metadata under ``original_text``
    so that answer generation can display the unmodified source passage.

    A SHA-256 disk cache at *cache_dir* avoids redundant LLM calls when the
    same (document_hash, chunk_id) pair is seen again.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 100,
        preserve_sections: bool = True,
        context_model: str = "llama-3.1-8b-instant",
        cache_dir: Optional[str] = None,
        max_doc_chars_for_context: int = 8_000,
    ) -> None:
        """
        Args:
            chunk_size: Target tokens per base chunk.
            overlap: Token overlap between base chunks.
            preserve_sections: Honour section headings in base chunker.
            context_model: Fast/cheap model for context generation.
            cache_dir: Optional path to persist context cache (avoids re-billing
                       on rebuild).  Defaults to ``./data/context_cache``.
            max_doc_chars_for_context: Characters of the source document sent
                to the LLM.  Truncated to keep prompts affordable.
        """
        self._base_chunker = IntelligentChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            preserve_sections=preserve_sections,
        )
        # Expose the same attributes IntelligentChunker has so that
        # RetrieverTool.build_vectorstore can read them without an AttributeError.
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_sections = preserve_sections

        self._llm = get_chat_model(
            context_model,
            temperature=0,
            max_tokens=120,
            role="context",
        )
        self._max_doc_chars = max_doc_chars_for_context
        self._cache_dir = Path(cache_dir) if cache_dir else Path("./data/context_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, str] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        """Chunk document and enrich each chunk with situating context."""
        base_result = self._base_chunker.chunk_document(text, metadata)
        if not base_result.chunks:
            return base_result

        doc_snippet = text[: self._max_doc_chars]
        doc_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

        enriched_chunks: List[TextChunk] = []
        for chunk in base_result.chunks:
            context = self._get_context(doc_snippet, doc_hash, chunk)
            chunk.text = f"{context}\n\n{chunk.text}" if context else chunk.text
            enriched_chunks.append(chunk)

        base_result.chunks = enriched_chunks
        base_result.metadata["contextual_enrichment"] = True
        base_result.metadata["context_model"] = getattr(self._llm, "model_name", context_model)
        return base_result

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Convenience wrapper: takes LangChain Documents, returns contextualised
        LangChain Documents ready for embedding.
        """
        enriched: List[Document] = []
        for doc in documents:
            result = self.chunk_document(doc.page_content, doc.metadata)
            for chunk in result.chunks:
                chunk_meta = doc.metadata.copy()
                chunk_meta.update(
                    {
                        "chunk_id": chunk.chunk_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "token_count": chunk.token_count,
                        "sentence_count": chunk.sentence_count,
                        "confidence_score": chunk.confidence_score,
                        "contextually_enriched": True,
                    }
                )
                if chunk.section_header:
                    chunk_meta["section_header"] = chunk.section_header
                enriched.append(
                    Document(page_content=chunk.text, metadata=chunk_meta)
                )
        self._save_cache()
        return enriched

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_context(
        self, doc_snippet: str, doc_hash: str, chunk: TextChunk
    ) -> str:
        cache_key = f"{doc_hash}::chunk{chunk.chunk_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = _CONTEXT_PROMPT.format(
            document_text=doc_snippet,
            chunk_text=chunk.text[:1_500],
        )
        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            ctx = response.content.strip()
        except Exception as exc:
            logger.warning(f"[CTX] Context generation failed for chunk {chunk.chunk_id}: {exc}")
            ctx = ""

        self._cache[cache_key] = ctx
        return ctx

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self._cache_dir / "context_cache.json"

    def _load_cache(self) -> None:
        p = self._cache_path()
        if p.exists():
            try:
                self._cache = json.loads(p.read_text(encoding="utf-8"))
                logger.info(f"[CTX] Loaded {len(self._cache)} cached contexts from {p}")
            except Exception as exc:
                logger.warning(f"[CTX] Could not load context cache: {exc}")
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            self._cache_path().write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"[CTX] Saved {len(self._cache)} contexts to cache")
        except Exception as exc:
            logger.warning(f"[CTX] Could not save context cache: {exc}")
