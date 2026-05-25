"""
RAPTOR — Recursive Abstractive Processing for Tree-Organised Retrieval.

Reference: Sarthi et al. "RAPTOR: Recursive Abstractive Processing for
           Tree-Organised Retrieval" (ICLR 2024).
           https://arxiv.org/abs/2401.18059

Architecture overview
---------------------
Level 0 (leaves):  raw chunks from ContextualChunker / IntelligentChunker
Level 1:  LLM summaries of semantically-similar L0 clusters
Level 2:  LLM summaries of L1 clusters (optional, for large corpora)
…
Level N:  single root summary (corpus-wide)

All levels are stored in ChromaDB with metadata ``raptor_level``.  At query
time, a *collapsed-tree* search runs similarity search across all levels
simultaneously and the retriever returns whichever levels are most relevant —
high-level summaries answer broad questions while leaf chunks answer specific
numerical queries.

Clustering uses UMAP (dimensionality reduction) + Gaussian Mixture Models
(soft assignment).  If UMAP/sklearn are unavailable it falls back to
k-means-style centroid clustering using numpy only.
"""
from __future__ import annotations

import logging
import math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

from ..core.model_factory import get_chat_model, get_embedding_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SUMMARY_PROMPT = """\
You are an expert petrophysicist summarising Volve field documents.

Below are {n_chunks} related passages from one or more Volve well reports:

{passages}

Write a concise technical summary (4–6 sentences) that captures the key
petrophysical facts, well names, formations, and parameters mentioned across
all passages.  Be factual; do not invent values."""


# ---------------------------------------------------------------------------
# Pure-numpy fallback clustering
# ---------------------------------------------------------------------------

def _numpy_cluster(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Lightweight k-means using numpy (no sklearn required)."""
    n = embeddings.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n_clusters, n), replace=False)
    centroids = embeddings[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(20):
        dists = np.linalg.norm(embeddings[:, None] - centroids[None], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            members = embeddings[labels == k]
            if len(members):
                centroids[k] = members.mean(axis=0)
    return labels


def _cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int,
    use_umap: bool = True,
) -> np.ndarray:
    """Cluster embeddings; returns integer label array of shape (n,)."""
    if use_umap:
        try:
            import umap  # type: ignore
            from sklearn.mixture import GaussianMixture  # type: ignore

            reduced = umap.UMAP(
                n_components=min(10, embeddings.shape[1] - 1, embeddings.shape[0] - 1),
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ).fit_transform(embeddings)
            gm = GaussianMixture(n_components=n_clusters, random_state=42)
            return gm.fit_predict(reduced)
        except ImportError:
            logger.info("[RAPTOR] umap/sklearn not installed; using numpy k-means fallback")
        except Exception as exc:
            logger.warning(f"[RAPTOR] UMAP/GMM clustering failed: {exc}; using numpy fallback")
    return _numpy_cluster(embeddings, n_clusters)


# ---------------------------------------------------------------------------
# Main RAPTOR indexer
# ---------------------------------------------------------------------------

class RaptorIndexer:
    """
    Builds a multi-level RAPTOR tree from leaf chunks and returns a flat list
    of Documents at all levels, ready to be inserted into ChromaDB.

    Parameters
    ----------
    summary_model:   LLM used to summarise clusters.
    embedding_model: Embedding model name (must match RetrieverTool's).
    max_levels:      Maximum tree depth (2 is usually sufficient for ~1k chunks).
    target_clusters: Approximate number of clusters per level.
    min_cluster_size: Minimum number of chunks before splitting into clusters.
    max_tokens_per_summary: Maximum tokens in each cluster summary.
    """

    def __init__(
        self,
        summary_model: str = "llama-3.1-8b-instant",
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
        max_levels: int = 2,
        target_clusters: int = 10,
        min_cluster_size: int = 3,
        max_tokens_per_summary: int = 400,
    ) -> None:
        self._llm = get_chat_model(
            summary_model,
            temperature=0,
            max_tokens=max_tokens_per_summary,
            role="raptor",
        )
        self._embeddings = get_embedding_model(embedding_model)
        self._max_levels = max_levels
        self._target_clusters = target_clusters
        self._min_cluster_size = min_cluster_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_tree(self, leaf_docs: List[Document]) -> List[Document]:
        """
        Build the RAPTOR tree from leaf documents.

        Returns all documents (leaves + summaries at every level) with
        metadata enriched with ``raptor_level`` (0 = leaf).
        """
        if not leaf_docs:
            return []

        # Ensure leaves carry raptor_level = 0
        for doc in leaf_docs:
            doc.metadata.setdefault("raptor_level", 0)

        all_docs: List[Document] = list(leaf_docs)
        current_level_docs = leaf_docs

        for level in range(1, self._max_levels + 1):
            if len(current_level_docs) < self._min_cluster_size:
                logger.info(f"[RAPTOR] Level {level}: only {len(current_level_docs)} docs — stopping tree growth")
                break

            logger.info(f"[RAPTOR] Building level {level} from {len(current_level_docs)} docs…")
            summary_docs = self._build_level(current_level_docs, level)
            if not summary_docs:
                break
            all_docs.extend(summary_docs)
            current_level_docs = summary_docs
            logger.info(f"[RAPTOR] Level {level}: produced {len(summary_docs)} summary nodes")

        logger.info(f"[RAPTOR] Tree built: {len(all_docs)} total documents across {self._max_levels + 1} levels")
        return all_docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_level(
        self, docs: List[Document], level: int
    ) -> List[Document]:
        """Cluster docs, summarise each cluster, return summary Documents."""
        texts = [d.page_content for d in docs]
        embeddings = self._embed_texts(texts)
        if embeddings is None:
            return []

        n_clusters = self._choose_n_clusters(len(docs))
        labels = _cluster_embeddings(embeddings, n_clusters=n_clusters)

        # Group docs by cluster
        clusters: Dict[int, List[Document]] = {}
        for doc, label in zip(docs, labels):
            clusters.setdefault(int(label), []).append(doc)

        summaries: List[Document] = []
        for cluster_id, cluster_docs in clusters.items():
            if len(cluster_docs) < 1:
                continue
            summary_text = self._summarise_cluster(cluster_docs)
            if not summary_text:
                continue
            # Inherit representative metadata from the first doc in cluster
            meta = self._merge_metadata(cluster_docs, level, cluster_id)
            summaries.append(Document(page_content=summary_text, metadata=meta))

        return summaries

    def _choose_n_clusters(self, n_docs: int) -> int:
        """Determine number of clusters for a given pool size."""
        # Aim for target_clusters but cap to avoid single-doc clusters
        return max(2, min(self._target_clusters, math.ceil(n_docs / self._min_cluster_size)))

    def _embed_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed a list of texts and return a numpy matrix."""
        try:
            vecs = self._embeddings.embed_documents(
                [t[:6000] for t in texts]  # truncate to stay within token limits
            )
            return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            logger.error(f"[RAPTOR] Embedding failed: {exc}")
            return None

    def _summarise_cluster(self, docs: List[Document]) -> str:
        """Call LLM to summarise a cluster of documents."""
        passages = "\n\n---\n\n".join(d.page_content[:800] for d in docs[:8])
        prompt = _SUMMARY_PROMPT.format(n_chunks=len(docs), passages=passages)
        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as exc:
            logger.warning(f"[RAPTOR] Summarisation failed: {exc}")
            return ""

    def _merge_metadata(
        self, docs: List[Document], level: int, cluster_id: int
    ) -> Dict[str, Any]:
        """Build metadata for a summary node."""
        first = docs[0].metadata if docs else {}
        wells: set[str] = set()
        sources: set[str] = set()
        for d in docs:
            w = d.metadata.get("well_name", "")
            s = d.metadata.get("source", "")
            if w:
                wells.add(w)
            if s:
                sources.add(s)
        return {
            "raptor_level": level,
            "raptor_cluster_id": cluster_id,
            "raptor_n_children": len(docs),
            "raptor_wells": ", ".join(sorted(wells)),
            "raptor_sources": ", ".join(sorted(sources)),
            "document_type": first.get("document_type", "raptor_summary"),
            "is_raptor_summary": True,
        }
