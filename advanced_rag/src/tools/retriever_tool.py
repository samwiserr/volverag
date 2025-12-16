"""
Retriever tool for LangGraph agentic RAG.
Creates and manages ChromaDB vector store with OpenAI embeddings.
Uses IntelligentChunker for semantic boundary detection.
"""
import os
import logging
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from ..processors.intelligent_chunker import IntelligentChunker
from .cross_encoder_reranker import rerank_documents

logger = logging.getLogger(__name__)


class RetrieverTool:
    """Manages ChromaDB vector store and retriever for RAG."""
    
    def __init__(
        self,
        persist_directory: str = "./data/vectorstore",
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "petrophysical_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 150
    ):
        """
        Initialize retriever tool.
        
        Args:
            persist_directory: Directory to persist ChromaDB
            embedding_model: OpenAI embedding model name
            collection_name: ChromaDB collection name
            chunk_size: Target tokens per chunk (default: 500 for finer indexing)
            chunk_overlap: Token overlap between chunks (default: 150, ~30% overlap)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )
        
        # Initialize IntelligentChunker for semantic boundary detection
        self.chunker = IntelligentChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            preserve_sections=True
        )
        
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None

        # Hybrid lexical store (persisted)
        self.lexical_store_path = self.persist_directory / "lexical_store.jsonl"
        self.lexical_meta_path = self.persist_directory / "lexical_store_meta.json"
        self._lex_docs: List[Document] = []
        self._lex_tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        # Optional cross-encoder reranker (Phase 2)
        self._use_cross_encoder = os.getenv("RAG_USE_CROSS_ENCODER", "true").lower() in {"1", "true", "yes"}
        
        # Optional LLM reranker
        self._rerank_enabled = os.getenv("RAG_RERANK", "llm").lower() in {"1", "true", "yes", "llm"}
        self._rerank_model = os.getenv("RAG_RERANK_MODEL", "gpt-4o")
        self._reranker_llm = ChatOpenAI(model=self._rerank_model, temperature=0)

        # MMR diversification (embedding-based) over merged candidates before rerank
        self._mmr_enabled = os.getenv("RAG_MMR", "true").lower() in {"1", "true", "yes"}
        self._mmr_lambda = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))
        self._embed_cache: Dict[str, List[float]] = {}
        
        logger.info(f"[OK] RetrieverTool initialized with model: {embedding_model}")
        logger.info(f"[OK] Using IntelligentChunker: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def build_vectorstore(self, documents: List[Document], chunk_size: int = None, chunk_overlap: int = None):
        """
        Build ChromaDB vector store from documents using IntelligentChunker.
        
        Args:
            documents: List of LangChain Document objects
            chunk_size: Size of text chunks (uses instance default if None)
            chunk_overlap: Overlap between chunks (uses instance default if None)
        """
        logger.info(f"Building vector store from {len(documents)} documents...")
        
        # Use instance defaults if not provided
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        # Update chunker if parameters changed
        if chunk_size != self.chunker.chunk_size or chunk_overlap != self.chunker.overlap:
            self.chunker = IntelligentChunker(
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                preserve_sections=True
            )
        
        # Chunk documents using IntelligentChunker
        all_splits = []
        for doc in documents:
            try:
                # Use IntelligentChunker for semantic boundary detection
                chunking_result = self.chunker.chunk_document(
                    text=doc.page_content,
                    metadata=doc.metadata
                )
                
                # Convert TextChunk objects to LangChain Document objects
                for text_chunk in chunking_result.chunks:
                    # Preserve original document metadata
                    chunk_metadata = doc.metadata.copy()
                    
                    # Add chunk-specific metadata
                    chunk_metadata.update({
                        'chunk_id': text_chunk.chunk_id,
                        'start_char': text_chunk.start_char,
                        'end_char': text_chunk.end_char,
                        'token_count': text_chunk.token_count,
                        'sentence_count': text_chunk.sentence_count,
                        'confidence_score': text_chunk.confidence_score,
                    })
                    
                    # TOC detection (avoid returning dotted-leader TOC entries as "answers")
                    chunk_metadata["is_toc"] = self._is_toc_text(text_chunk.text)
                    
                    # Add section header if available
                    if text_chunk.section_header:
                        chunk_metadata['section_header'] = text_chunk.section_header
                    
                    # Create LangChain Document
                    chunk_doc = Document(
                        page_content=text_chunk.text,
                        metadata=chunk_metadata
                    )
                    all_splits.append(chunk_doc)
                    
            except Exception as e:
                logger.warning(f"Failed to chunk document {doc.metadata.get('filename', 'unknown')}: {e}")
                # Fallback: create a single chunk from the document
                all_splits.append(doc)
        
        splits = all_splits
        
        # Ensure metadata is preserved in all chunks
        for split in splits:
            # Ensure TOC metadata exists
            if "is_toc" not in split.metadata:
                split.metadata["is_toc"] = self._is_toc_text(split.page_content)

            # Preserve important metadata fields
            if 'is_well_picks' not in split.metadata:
                # Inherit from source document if available
                source = split.metadata.get('source', '')
                if 'Well_picks' in source or 'well_picks' in source.lower():
                    split.metadata['is_well_picks'] = True
                    split.metadata['is_formation_data'] = True
                    split.metadata['document_type'] = 'well_picks'
        
        logger.info(f"Split into {len(splits)} chunks using IntelligentChunker")
        well_picks_count = sum(1 for s in splits if s.metadata.get('is_well_picks'))
        if well_picks_count > 0:
            logger.info(f"  - {well_picks_count} chunks from well picks document")
        
        # Log chunking statistics
        if splits:
            avg_tokens = sum(s.metadata.get('token_count', 0) for s in splits) / len(splits)
            sections_with_headers = sum(1 for s in splits if s.metadata.get('section_header'))
            logger.info(f"  - Average tokens per chunk: {avg_tokens:.1f}")
            logger.info(f"  - Chunks with section headers: {sections_with_headers}")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name
        )
        
        # Create retriever with more documents for better recall
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve more documents for better context
        )

        # Persist lexical store for BM25 hybrid retrieval
        self._persist_lexical_store(splits)
        self._load_lexical_store()  # build bm25 in-memory for this process
        
        logger.info(f"[OK] Vector store built with {len(splits)} chunks")
    
    def load_vectorstore(self) -> bool:
        """
        Load existing vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not (self.persist_directory / "chroma.sqlite3").exists():
                logger.warning("Vector store not found. Run build_index first.")
                return False
            
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}  # Retrieve more documents for better context
            )

            # Load lexical store if available; if not, bootstrap from Chroma
            if not self._load_lexical_store():
                self._bootstrap_lexical_store_from_chroma()
                self._load_lexical_store()
            
            logger.info("[OK] Vector store loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    def _tokenize(self, text: str) -> List[str]:
        # Lightweight tokenization for petrophysical docs
        # Keep numbers and well/formation punctuation tokens like 15/9-F-11.
        return re.findall(r"[a-z0-9]+(?:[/-][a-z0-9]+)*", text.lower())

    def _expand_query(self, query: str) -> List[str]:
        """
        Field-aware query expansion:
        - Well-name normalization/variants (15/9-F-11 == 15_9-F-11 == 15-9-F-11)
        - Acronym expansion (OWC, ODT, RKB, etc.)
        """
        q = query.strip()
        ql = q.lower()

        acronyms = {
            "owc": "oil-water contact",
            "odt": "oil down to",
            "goc": "gas-oil contact",
            "rkb": "rotary kelly bushing",
            "md": "measured depth",
            "tvd": "true vertical depth",
            "tvdss": "true vertical depth sub sea",
            "wlc": "well log correlation",
            "lfp": "log formation parameters",
        }

        expanded_terms: List[str] = []
        for ac, full in acronyms.items():
            if re.search(rf"\b{re.escape(ac)}\b", ql):
                expanded_terms.append(full)

        # Well-name variants
        # Capture patterns like 15/9-F-11, 15_9-F-11, 15-9-F-11, 15/9-19A, etc.
        m = re.search(r"(15[\s_/-]*9[\s_/-]*(?:f[\s_/-]*)?\d+[a-z]?(?:[\s_/-]*t2)?)", q, re.IGNORECASE)
        well_variants: List[str] = []
        if m:
            well_raw = m.group(1)
            # Normalize components
            well_clean = re.sub(r"\s+", "", well_raw)
            # Build variants for separators
            variants = set()
            variants.add(well_clean.replace("_", "/").replace("-", "/"))
            variants.add(well_clean.replace("/", "_").replace("-", "_"))
            variants.add(well_clean.replace("/", "-").replace("_", "-"))
            # Keep original too
            variants.add(well_raw)
            # Also add with "Well " prefix commonly seen
            for v in list(variants):
                variants.add(f"Well {v}")
            well_variants = sorted(variants)

        queries = [q]
        if expanded_terms:
            # Add a query that includes expansions (keeps original wording too)
            queries.append(q + " " + " ".join(expanded_terms))
        for v in well_variants[:6]:
            # avoid blowing up query count
            if v and v not in q:
                queries.append(q.replace(m.group(1), v) if m else q + " " + v)

        # Deduplicate while preserving order
        out: List[str] = []
        seen = set()
        for qq in queries:
            key = qq.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(qq.strip())
        return out[:6]

    def _is_toc_text(self, text: str) -> bool:
        """Heuristic TOC detection: dotted leaders and many short heading+page lines."""
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return False
        dotted = sum(1 for ln in lines if re.search(r"\.{4,}\s*\d+\s*$", ln))
        short_pg = sum(1 for ln in lines if len(ln) < 60 and re.search(r"\b\d+\s*$", ln))
        if dotted >= 2:
            return True
        if dotted >= 1 and (dotted + short_pg) / max(len(lines), 1) > 0.4:
            return True
        return False

    def _doc_key(self, doc: Document) -> str:
        src = str(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", doc.metadata.get("page_number", "")))
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        h = hashlib.md5()
        h.update((src + "|" + page + "|" + chunk_id + "|" + doc.page_content[:2000]).encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _cosine(self, a: List[float], b: List[float]) -> float:
        # Manual cosine to avoid extra deps
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    def _embed_text_cached(self, key: str, text: str) -> List[float]:
        if key in self._embed_cache:
            return self._embed_cache[key]
        # Use embeddings model to embed short snippets
        vec = self.embeddings.embed_query(text[:2000])
        self._embed_cache[key] = vec
        return vec

    def _mmr_select(self, query: str, docs: List[Document], k: int, lambda_mult: float) -> List[Document]:
        """
        Classic MMR:
        pick doc maximizing lambda*sim(query, doc) - (1-lambda)*max_sim(doc, selected)
        """
        if not docs or k <= 0:
            return []
        if len(docs) <= k:
            return docs

        q_emb = self._embed_text_cached(f"q::{hashlib.md5(query.encode('utf-8', errors='ignore')).hexdigest()}", query)
        doc_embs: List[List[float]] = []
        doc_keys: List[str] = []
        for d in docs:
            dk = d.metadata.get("lexical_id") or self._doc_key(d)
            doc_keys.append(dk)
            doc_embs.append(self._embed_text_cached(f"d::{dk}", d.page_content))

        # Precompute relevance to query
        rel = [self._cosine(q_emb, e) for e in doc_embs]

        selected: List[int] = []
        candidates = list(range(len(docs)))

        # Start with best relevance
        first = max(candidates, key=lambda i: rel[i])
        selected.append(first)
        candidates.remove(first)

        while candidates and len(selected) < k:
            def mmr_score(i: int) -> float:
                max_sim = 0.0
                for j in selected:
                    sim = self._cosine(doc_embs[i], doc_embs[j])
                    if sim > max_sim:
                        max_sim = sim
                return lambda_mult * rel[i] - (1.0 - lambda_mult) * max_sim

            nxt = max(candidates, key=mmr_score)
            selected.append(nxt)
            candidates.remove(nxt)

        return [docs[i] for i in selected]

    def _persist_lexical_store(self, docs: List[Document]) -> None:
        try:
            # Write JSONL where each line is {"id","text","metadata","tokens"}
            with self.lexical_store_path.open("w", encoding="utf-8") as f:
                for d in docs:
                    did = self._doc_key(d)
                    tokens = self._tokenize(d.page_content)
                    payload = {"id": did, "text": d.page_content, "metadata": d.metadata, "tokens": tokens}
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

            meta = {"count": len(docs)}
            self.lexical_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info(f"[OK] Persisted lexical store with {len(docs)} docs")
        except Exception as e:
            logger.warning(f"[LEXICAL] Failed to persist lexical store: {e}")

    def _load_lexical_store(self) -> bool:
        if not self.lexical_store_path.exists():
            return False
        try:
            docs: List[Document] = []
            tokens: List[List[str]] = []
            with self.lexical_store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    meta = obj.get("metadata", {}) or {}
                    meta["lexical_id"] = obj.get("id")
                    docs.append(Document(page_content=text, metadata=meta))
                    tokens.append(obj.get("tokens") or self._tokenize(text))
            self._lex_docs = docs
            self._lex_tokens = tokens
            self._bm25 = BM25Okapi(tokens) if tokens else None
            logger.info(f"[OK] Loaded lexical store with {len(docs)} docs")
            return True
        except Exception as e:
            logger.warning(f"[LEXICAL] Failed to load lexical store: {e}")
            return False

    def _bootstrap_lexical_store_from_chroma(self) -> None:
        """If lexical store is missing, rebuild it from Chroma collection contents."""
        try:
            if not self.vectorstore or not hasattr(self.vectorstore, "_collection") or not self.vectorstore._collection:
                return
            results = self.vectorstore._collection.get(limit=100000)
            docs: List[Document] = []
            if results and results.get("documents"):
                for i, text in enumerate(results["documents"]):
                    meta = {}
                    if results.get("metadatas") and i < len(results["metadatas"]):
                        meta = results["metadatas"][i] or {}
                    docs.append(Document(page_content=text, metadata=meta))
            if docs:
                self._persist_lexical_store(docs)
        except Exception as e:
            logger.warning(f"[LEXICAL] Bootstrap from Chroma failed: {e}")

    def _bm25_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        if not self._bm25 or not self._lex_docs:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        # Get top k indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: List[Tuple[Document, float]] = []
        for i in ranked:
            out.append((self._lex_docs[i], float(scores[i])))
        return out

    def _vector_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            return []
        try:
            pairs = self.vectorstore.similarity_search_with_score(query, k=k)
            # Score is distance; convert to similarity-ish
            out: List[Tuple[Document, float]] = []
            for d, dist in pairs:
                sim = 1.0 / (1.0 + float(dist))
                out.append((d, sim))
            return out
        except Exception:
            # fallback to retriever rank only
            docs = self.vectorstore.similarity_search(query, k=k)
            return [(d, 1.0 / (1.0 + idx)) for idx, d in enumerate(docs)]

    class _RerankOut(BaseModel):
        ordered_indices: List[int] = Field(description="Indices of documents in best-to-worst order")

    def _llm_rerank(self, query: str, docs: List[Document], top_n: int = 12) -> List[Document]:
        if not self._rerank_enabled or not docs:
            return docs
        # Limit cost
        cand = docs[: min(len(docs), 24)]
        # Provide short excerpts
        items = []
        for i, d in enumerate(cand):
            src = d.metadata.get("source", d.metadata.get("filename", ""))
            page = d.metadata.get("page", d.metadata.get("page_number", ""))
            excerpt = d.page_content[:500].replace("\n", " ")
            items.append(f"[{i}] source={src} page={page} text={excerpt}")
        prompt = (
            "You are reranking candidate passages for a question.\n"
            "Return the indices of the best passages to answer the question, best-to-worst.\n"
            "Only return a JSON object with key 'ordered_indices'.\n\n"
            f"Question: {query}\n\n"
            "Candidates:\n" + "\n".join(items)
        )
        try:
            resp = self._reranker_llm.with_structured_output(self._RerankOut).invoke([{"role": "user", "content": prompt}])
            order = [i for i in resp.ordered_indices if isinstance(i, int) and 0 <= i < len(cand)]
            if not order:
                return docs
            reranked = [cand[i] for i in order]
            # Append any not mentioned, preserving original order
            seen = set(order)
            for i in range(len(cand)):
                if i not in seen:
                    reranked.append(cand[i])
            # Replace front of docs with reranked subset
            rest = docs[len(cand):]
            return reranked[:top_n] + rest
        except Exception as e:
            logger.warning(f"[RERANK] LLM rerank failed: {e}")
            return docs

    def _hybrid_retrieve(self, queries: Iterable[str], k_vec: int = 20, k_lex: int = 30, k_final: int = 10) -> List[Document]:
        # Run hybrid over expanded queries and merge
        all_vec: List[Tuple[Document, float]] = []
        all_lex: List[Tuple[Document, float]] = []
        for q in queries:
            all_vec.extend(self._vector_search(q, k=k_vec))
            all_lex.extend(self._bm25_search(q, k=k_lex))

        # Normalize BM25 scores to [0,1] using max
        max_bm = max((s for _d, s in all_lex), default=0.0)
        # Merge with weighted score
        merged: Dict[str, Tuple[Document, float]] = {}

        for idx, (d, s) in enumerate(all_vec):
            key = self._doc_key(d)
            merged[key] = (d, 0.65 * s + 0.35 * (1.0 / (1.0 + idx)))

        for idx, (d, s) in enumerate(all_lex):
            key = d.metadata.get("lexical_id") or self._doc_key(d)
            norm = (s / max_bm) if max_bm > 0 else 0.0
            prev = merged.get(key)
            score = 0.55 * norm + 0.45 * (1.0 / (1.0 + idx))
            if prev is None or score > prev[1]:
                merged[key] = (d, max(prev[1], score) if prev else score)

        ranked = sorted(merged.values(), key=lambda t: t[1], reverse=True)
        docs = [d for d, _s in ranked][: max(k_final, 24)]

        # Filter TOC-like chunks unless explicitly asked for TOC/pages
        q0 = next(iter(queries))
        q0l = (q0 or "").lower()
        wants_toc = any(k in q0l for k in ["table of contents", "toc", "page", "pages"])
        if not wants_toc:
            docs = [d for d in docs if not d.metadata.get("is_toc")]

        # MMR diversification before rerank
        if self._mmr_enabled and docs:
            mmr_k = min(max(k_final, 10), len(docs))
            docs = self._mmr_select(next(iter(queries)), docs, k=mmr_k, lambda_mult=self._mmr_lambda) + docs
            # de-dup preserving order
            seen = set()
            deduped = []
            for d in docs:
                key = d.metadata.get("lexical_id") or self._doc_key(d)
                if key not in seen:
                    seen.add(key)
                    deduped.append(d)
            docs = deduped[: max(k_final, 24)]

        # Phase 2: Cross-encoder reranking before LLM rerank
        # Architecture: Hybrid → MMR → Cross-Encoder (top 24) → LLM Rerank (top 12) → Final
        query = next(iter(queries))
        if self._use_cross_encoder and len(docs) > 1:
            try:
                # Cross-encoder rerank top 24 candidates
                cross_encoder_k = min(24, len(docs))
                docs = rerank_documents(query, docs[:cross_encoder_k], top_k=cross_encoder_k) + docs[cross_encoder_k:]
                logger.info(f"[RETRIEVE] Cross-encoder reranked {cross_encoder_k} documents")
            except Exception as e:
                logger.warning(f"[RETRIEVE] Cross-encoder reranking failed, continuing with LLM rerank: {e}")

        docs = self._llm_rerank(query, docs, top_n=k_final)
        return docs[:k_final]
    
    def _normalize_well_name(self, well_name: str) -> str:
        """Normalize well name to handle different formats (15/9-19A = 15_9-19A = 15-9-19A)."""
        import re
        # Remove spaces, normalize separators to a common format
        normalized = re.sub(r'[\s_/-]', '', well_name.upper())
        return normalized
    
    def _retrieve_all_chunks_from_documents(self, document_sources: List[str]) -> List[Document]:
        """
        Retrieve ALL chunks from specific document sources.
        
        Args:
            document_sources: List of document source paths to retrieve all chunks from
            
        Returns:
            List of all Document chunks from the specified sources
        """
        all_chunks = []
        if not document_sources:
            return all_chunks
        
        try:
            if hasattr(self.vectorstore, '_collection') and self.vectorstore._collection:
                collection = self.vectorstore._collection
                
                # Retrieve all chunks for each document source
                for source in document_sources:
                    if not source:
                        continue
                    try:
                        # Get all chunks from this document
                        results = collection.get(
                            where={"source": source},
                            limit=10000  # Very high limit to get all chunks
                        )
                        
                        if results and 'documents' in results and results['documents']:
                            for i, doc_text in enumerate(results['documents']):
                                metadata = {}
                                if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas']):
                                    metadata = results['metadatas'][i] or {}
                                
                                # Create Document object
                                chunk_doc = Document(
                                    page_content=doc_text,
                                    metadata=metadata
                                )
                                all_chunks.append(chunk_doc)
                            
                            logger.info(f"[RETRIEVE] Retrieved {len(results['documents'])} chunks from document: {source}")
                    except Exception as e:
                        logger.warning(f"[RETRIEVE] Failed to retrieve chunks from {source}: {e}")
                        continue
                
                logger.info(f"[RETRIEVE] Retrieved total {len(all_chunks)} chunks from {len(document_sources)} document(s)")
                return all_chunks
            else:
                logger.warning("[RETRIEVE] Collection not accessible for full document retrieval")
                return []
        except Exception as e:
            logger.warning(f"[RETRIEVE] Error retrieving full documents: {e}")
            return []
    
    def _extract_well_name(self, query: str) -> Optional[str]:
        """Extract well name from query."""
        import re
        # Pattern: 15/9-19A, 15_9-19A, 15-9-19A, etc.
        # Try full well name first (15/9-19A)
        patterns = [
            r'(15[_\s/-]9[_\s/-]?\d+[A-Z]?)',  # 15/9-19A format (full well name)
            r'well\s+(15[_\s/-]9[_\s/-]?\d+[A-Z]?)',  # "well 15/9-19A"
            r'(15[_\s/-]9[_\s/-]?[-]?\d+[A-Z]?)',  # 15/9-19A with optional dash
            r'(\d+[_\s/-]\d+[_\s/-]\d+[A-Z]?)',  # General pattern (e.g., 15/9-19A)
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Return the full match, prioritizing groups if they exist
                well_name = match.group(1) if match.lastindex else match.group(0)
                # Normalize separators to a consistent format
                well_name = re.sub(r'[\s_]', '/', well_name)  # Convert spaces/underscores to /
                return well_name
        return None
    
    def _filter_docs_by_well(self, docs: List[Document], well_name: str) -> List[Document]:
        """Filter documents to only those containing the specified well name."""
        normalized_query_well = self._normalize_well_name(well_name)
        filtered_docs = []
        
        for doc in docs:
            doc_text = doc.page_content.upper()
            doc_source = doc.metadata.get('source', '').upper()
            doc_filename = doc.metadata.get('filename', '').upper()
            
            # Check if document contains the well name in various formats
            well_variants = [
                well_name.upper(),
                f"15/9-{well_name}",
                f"15_9-{well_name}",
                f"15-9-{well_name}",
                f"15/9-{well_name.replace('A', '')}A",  # Handle A suffix variations
                f"NO 15/9-{well_name}",
                f"NO_15/9-{well_name}",
                f"NO 15_9-{well_name}",
                f"NO 15-9-{well_name}",
            ]
            
            # Check normalized well name
            doc_normalized = self._normalize_well_name(doc_text)
            if (normalized_query_well in doc_normalized or
                any(variant in doc_text for variant in well_variants) or
                any(variant in doc_source for variant in well_variants) or
                any(variant in doc_filename for variant in well_variants)):
                filtered_docs.append(doc)
        
        return filtered_docs
        
        @tool
        def retrieve_petrophysical_docs(query: str) -> str:
            """Search and return information from petrophysical documents.
            
            Use this tool to retrieve relevant context from the document database
            when answering questions about wells, formations, petrophysical data,
            or any information contained in the processed documents.
            
            For queries about "all wells" or "all formations", this will retrieve
            information from multiple documents to provide comprehensive answers.
            The well picks document is automatically prioritized for formation queries.
            
            Args:
                query: The search query to find relevant documents
                
            Returns:
                Concatenated text from relevant document chunks
            """
            query_lower = query.lower()
            
            # Extract well name from query for filtering
            well_name = self._extract_well_name(query)
            if well_name:
                logger.info(f"[RETRIEVE] Detected well name in query: {well_name}")
            
            # Detect formation-related queries
            is_formation_query = any(term in query_lower for term in [
                "formation", "formations", "all wells", "all formations", 
                "well picks", "formation picks", "formation data",
                "what formations", "list formations", "formations in"
            ])
            
            # Detect depth-related queries
            is_depth_query = any(term in query_lower for term in [
                "depth", "depths", "md", "tvd", "tvdss", "measured depth",
                "true vertical depth", "at what depth", "depth of"
            ])
            
            # Detect if query needs information from multiple sources
            needs_multiple = any(term in query_lower for term in ["all", "every", "each", "list", "summary", "overview"])
            
            # Detect comprehensive list queries that need ALL data
            is_comprehensive_list = any(term in query_lower for term in ["list", "each", "all", "every"]) and is_formation_query
            
            all_docs = []
            
            # For formation or depth queries, prioritize well picks document
            if is_formation_query or is_depth_query:
                try:
                    # For comprehensive list queries, get ALL well picks chunks directly
                    if is_comprehensive_list:
                        # Try to get ALL well picks chunks by fetching from collection directly
                        try:
                            # Access ChromaDB collection directly to get all well picks chunks
                            # ChromaDB LangChain wrapper exposes collection via _collection
                            if hasattr(self.vectorstore, '_collection') and self.vectorstore._collection:
                                collection = self.vectorstore._collection
                                # Get all documents with well picks metadata
                                results = collection.get(
                                    where={"is_well_picks": True},
                                    limit=10000  # Very high limit to get all
                                )
                                # Convert to Document objects
                                well_picks_docs = []
                                if results and 'documents' in results and results['documents']:
                                    for i, doc_text in enumerate(results['documents']):
                                        metadata = {}
                                        if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas']):
                                            metadata = results['metadatas'][i] or {}
                                        well_picks_docs.append(Document(
                                            page_content=doc_text,
                                            metadata=metadata
                                        ))
                                if well_picks_docs:
                                    # For comprehensive lists, don't filter by well name (user wants all wells)
                                    # But if a specific well is mentioned, we still want to filter
                                    if well_name and not is_comprehensive_list:
                                        original_count = len(well_picks_docs)
                                        well_picks_docs = self._filter_docs_by_well(well_picks_docs, well_name)
                                        if well_picks_docs:
                                            logger.info(f"[RETRIEVE] Filtered comprehensive list to {len(well_picks_docs)} chunks for well {well_name} (from {original_count})")
                                    all_docs.extend(well_picks_docs)
                                    logger.info(f"[RETRIEVE] Retrieved ALL {len(well_picks_docs)} well picks chunks directly from collection")
                            else:
                                raise AttributeError("Collection not accessible")
                        except Exception as e:
                            logger.warning(f"[RETRIEVE] Direct collection access failed: {e}, using similarity search with high k")
                            # Fallback: use similarity search with very high k and generic query
                            k_well_picks = 500
                            try:
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={
                                        "k": k_well_picks,
                                        "filter": {"is_well_picks": True}
                                    }
                                )
                                well_picks_docs = well_picks_retriever.invoke("formation picks well")
                            except:
                                # Last resort: no filter, just high k
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": k_well_picks}
                                )
                                well_picks_docs = well_picks_retriever.invoke("formation picks well")
                                # Filter manually
                                well_picks_docs = [doc for doc in well_picks_docs 
                                                  if doc.metadata.get('is_well_picks') or 
                                                  'Well_picks' in doc.metadata.get('source', '') or
                                                  'Well_picks' in doc.metadata.get('filename', '')]
                            
                            if well_picks_docs:
                                all_docs.extend(well_picks_docs)
                                logger.info(f"[RETRIEVE] Found {len(well_picks_docs)} well picks chunks via similarity search")
                    else:
                        # Regular formation query - use similarity search
                        k_well_picks = 100 if needs_multiple else 15
                        
                        # First, try to get well picks chunks using metadata filtering
                        try:
                            # Method 1: Direct metadata filter (ChromaDB native)
                            well_picks_retriever = self.vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={
                                    "k": k_well_picks,
                                    "filter": {"is_well_picks": {"$eq": True}}  # ChromaDB filter syntax
                                }
                            )
                            well_picks_docs = well_picks_retriever.invoke(query)
                        except:
                            # Method 2: Try boolean filter
                            try:
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={
                                        "k": k_well_picks,
                                        "filter": {"is_well_picks": True}
                                    }
                                )
                                well_picks_docs = well_picks_retriever.invoke(query)
                            except:
                                # Method 3: Get all well picks chunks by searching with high k
                                well_picks_query = "formation picks well" if needs_multiple else query
                                general_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": k_well_picks * 2}
                                )
                                general_docs = general_retriever.invoke(well_picks_query)
                                well_picks_docs = [doc for doc in general_docs 
                                                  if doc.metadata.get('is_well_picks') or 
                                                  doc.metadata.get('is_well_picks') == True or
                                                  'Well_picks' in doc.metadata.get('source', '') or
                                                  'Well_picks' in doc.metadata.get('filename', '')]
                        
                    if well_picks_docs:
                        # Filter by well name if specified
                        if well_name:
                            original_count = len(well_picks_docs)
                            well_picks_docs = self._filter_docs_by_well(well_picks_docs, well_name)
                            if well_picks_docs:
                                logger.info(f"[RETRIEVE] Filtered to {len(well_picks_docs)} chunks for well {well_name} (from {original_count})")
                            else:
                                logger.warning(f"[RETRIEVE] No chunks found for well {well_name} after filtering, using all {original_count} chunks")
                                # Re-fetch without filtering if filtering removed all results
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": k_well_picks}
                                )
                                well_picks_docs = well_picks_retriever.invoke(query)
                        
                        all_docs.extend(well_picks_docs)
                        logger.info(f"[RETRIEVE] Found {len(well_picks_docs)} well picks chunks for formation query")
                except Exception as e:
                    logger.warning(f"[RETRIEVE] Well picks retrieval failed, using fallback: {e}")
                    # Fallback: search for well picks by filename in results
                    try:
                        # Get many documents to filter from
                        general_retriever = self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 100}
                        )
                        general_docs = general_retriever.invoke(query)
                        well_picks_docs = [doc for doc in general_docs 
                                          if doc.metadata.get('is_well_picks') or 
                                          doc.metadata.get('is_well_picks') == True or
                                          'Well_picks' in doc.metadata.get('source', '') or
                                          'Well_picks' in doc.metadata.get('filename', '')]
                        if well_picks_docs:
                            all_docs.extend(well_picks_docs)
                            logger.info(f"[RETRIEVE] Found {len(well_picks_docs)} well picks chunks via fallback")
                    except Exception as e2:
                        logger.warning(f"[RETRIEVE] Fallback also failed: {e2}")
            
            # Get general results (if not already got well picks, or need more)
            if not all_docs or needs_multiple:
                try:
                    if needs_multiple:
                        general_retriever = self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 20}
                        )
                        general_docs = general_retriever.invoke(query)
                    else:
                        general_docs = self.retriever.invoke(query)
                    
                    # Add general docs, avoiding duplicates
                    existing_sources = {doc.metadata.get('source', '') for doc in all_docs}
                    for doc in general_docs:
                        if doc.metadata.get('source', '') not in existing_sources:
                            all_docs.append(doc)
                            existing_sources.add(doc.metadata.get('source', ''))
                except Exception as e:
                    logger.warning(f"[RETRIEVE] General retrieval failed: {e}")
            
            # If still no docs, use default retriever
            if not all_docs:
                docs = self.retriever.invoke(query)
                all_docs = docs
            
            # DOCUMENT-LEVEL RETRIEVAL: Retrieve ALL chunks from relevant documents
            # Identify unique document sources from initial retrieval
            relevant_sources = set()
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                if source:
                    relevant_sources.add(source)
            
            if relevant_sources:
                logger.info(f"[RETRIEVE] Identified {len(relevant_sources)} relevant document(s), retrieving ALL chunks")
                # Retrieve all chunks from each relevant document
                full_document_chunks = self._retrieve_all_chunks_from_documents(list(relevant_sources))
                
                if full_document_chunks:
                    # Replace initial chunks with full document chunks
                    # This ensures we have complete document context, not just top-k chunks
                    all_docs = full_document_chunks
                    logger.info(f"[RETRIEVE] Retrieved ALL chunks from {len(relevant_sources)} document(s) - total {len(all_docs)} chunks")
                else:
                    logger.warning("[RETRIEVE] Full document retrieval failed, using initial retrieval results")
            
            # For comprehensive queries, we already have all chunks, so no limit needed
            # But keep the logic for backward compatibility and logging
            if is_comprehensive_list:
                logger.info("[RETRIEVE] Comprehensive list query - using ALL chunks from relevant documents")
            elif needs_multiple and is_formation_query:
                logger.info("[RETRIEVE] Multiple source query - using ALL chunks from relevant documents")
            elif needs_multiple:
                logger.info("[RETRIEVE] Multiple source query - using ALL chunks from relevant documents")
            else:
                logger.info(f"[RETRIEVE] Single query - using ALL chunks from {len(relevant_sources)} relevant document(s)")
            
            # No max_docs limit - we want ALL chunks from relevant documents
            logger.info(f"[RETRIEVE] Returning {len(all_docs)} document chunks (full documents)")
            
            # Phase 1.5: Include source and page information in output for better citation
            formatted_chunks = []
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                page = doc.metadata.get('page') or doc.metadata.get('page_number')
                page_start = doc.metadata.get('page_start')
                page_end = doc.metadata.get('page_end')
                
                # Build citation string
                citation_parts = []
                if source:
                    citation_parts.append(f"Source: {source}")
                if page_start is not None and page_end is not None:
                    if page_start == page_end:
                        citation_parts.append(f"(page {page_start})")
                    else:
                        citation_parts.append(f"(pages {page_start}-{page_end})")
                elif page is not None:
                    citation_parts.append(f"(page {page})")
                
                citation = " ".join(citation_parts) if citation_parts else ""
                
                # Format chunk with citation
                if citation:
                    formatted_chunks.append(f"{doc.page_content}\n[{citation}]")
                else:
                    formatted_chunks.append(doc.page_content)
            
            return "\n\n".join(formatted_chunks)
        
        return retrieve_petrophysical_docs
    
    def get_retriever_tool(self):
        """
        Get the retriever tool for LangGraph agent.
        
        Returns:
            LangChain tool function
        """
        if not self.retriever:
            raise RuntimeError("Retriever not initialized. Call build_vectorstore() or load_vectorstore() first.")
        
        @tool
        def retrieve_petrophysical_docs(query: str) -> str:
            """Search and return information from petrophysical documents.
            
            Use this tool to retrieve relevant context from the document database
            when answering questions about wells, formations, petrophysical data,
            or any information contained in the processed documents.
            
            For queries about "all wells" or "all formations", this will retrieve
            information from multiple documents to provide comprehensive answers.
            The well picks document is automatically prioritized for formation queries.
            
            Args:
                query: The search query to find relevant documents
                
            Returns:
                Concatenated text from relevant document chunks
            """
            query_lower = query.lower()

            # Hybrid retrieval for general queries (BM25 + vector + rerank)
            # Skip when this is clearly a well-picks "all wells" query because that is handled by the structured tool.
            is_big_well_picks_list = (
                ("formation" in query_lower or "formations" in query_lower)
                and ("well" in query_lower or "wells" in query_lower)
                and any(k in query_lower for k in ["each", "every", "all", "complete", "entire"])
            )
            if not is_big_well_picks_list and self._bm25 is not None:
                expanded = self._expand_query(query)
                if len(expanded) > 1:
                    logger.info(f"[RETRIEVE] Query expanded into {len(expanded)} variants")
                hybrid_docs = self._hybrid_retrieve(expanded, k_vec=24, k_lex=40, k_final=10)
                if hybrid_docs:
                    logger.info(f"[RETRIEVE] Hybrid returning {len(hybrid_docs)} chunks")
                    # Phase 1.5: Include source and page information
                    formatted_chunks = []
                    for d in hybrid_docs:
                        source = d.metadata.get('source', '')
                        page = d.metadata.get('page') or d.metadata.get('page_number')
                        page_start = d.metadata.get('page_start')
                        page_end = d.metadata.get('page_end')
                        
                        citation_parts = []
                        if source:
                            citation_parts.append(f"Source: {source}")
                        if page_start is not None and page_end is not None:
                            if page_start == page_end:
                                citation_parts.append(f"(page {page_start})")
                            else:
                                citation_parts.append(f"(pages {page_start}-{page_end})")
                        elif page is not None:
                            citation_parts.append(f"(page {page})")
                        
                        citation = " ".join(citation_parts) if citation_parts else ""
                        if citation:
                            formatted_chunks.append(f"{d.page_content}\n[{citation}]")
                        else:
                            formatted_chunks.append(d.page_content)
                    return "\n\n".join(formatted_chunks)
            
            # Extract well name from query for filtering
            well_name = self._extract_well_name(query)
            if well_name:
                logger.info(f"[RETRIEVE] Detected well name in query: {well_name}")
            
            # Detect formation-related queries
            is_formation_query = any(term in query_lower for term in [
                "formation", "formations", "all wells", "all formations", 
                "well picks", "formation picks", "formation data",
                "what formations", "list formations", "formations in"
            ])
            
            # Detect depth-related queries
            is_depth_query = any(term in query_lower for term in [
                "depth", "depths", "md", "tvd", "tvdss", "measured depth",
                "true vertical depth", "at what depth", "depth of"
            ])
            
            # Detect if query needs information from multiple sources
            needs_multiple = any(term in query_lower for term in ["all", "every", "each", "list", "summary", "overview"])
            
            # Detect comprehensive list queries that need ALL data
            is_comprehensive_list = any(term in query_lower for term in ["list", "each", "all", "every"]) and is_formation_query
            
            all_docs = []
            
            # For formation or depth queries, prioritize well picks document
            if is_formation_query or is_depth_query:
                try:
                    # For comprehensive list queries, get ALL well picks chunks directly
                    if is_comprehensive_list:
                        # Try to get ALL well picks chunks by fetching from collection directly
                        try:
                            # Access ChromaDB collection directly to get all well picks chunks
                            # ChromaDB LangChain wrapper exposes collection via _collection
                            if hasattr(self.vectorstore, '_collection') and self.vectorstore._collection:
                                collection = self.vectorstore._collection
                                # Get all documents with well picks metadata
                                results = collection.get(
                                    where={"is_well_picks": True},
                                    limit=10000  # Very high limit to get all
                                )
                                # Convert to Document objects
                                well_picks_docs = []
                                if results and 'documents' in results and results['documents']:
                                    for i, doc_text in enumerate(results['documents']):
                                        metadata = {}
                                        if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas']):
                                            metadata = results['metadatas'][i] or {}
                                        well_picks_docs.append(Document(
                                            page_content=doc_text,
                                            metadata=metadata
                                        ))
                                if well_picks_docs:
                                    # For comprehensive lists, don't filter by well name (user wants all wells)
                                    # But if a specific well is mentioned, we still want to filter
                                    if well_name and not is_comprehensive_list:
                                        original_count = len(well_picks_docs)
                                        well_picks_docs = self._filter_docs_by_well(well_picks_docs, well_name)
                                        if well_picks_docs:
                                            logger.info(f"[RETRIEVE] Filtered comprehensive list to {len(well_picks_docs)} chunks for well {well_name} (from {original_count})")
                                    all_docs.extend(well_picks_docs)
                                    logger.info(f"[RETRIEVE] Retrieved ALL {len(well_picks_docs)} well picks chunks directly from collection")
                            else:
                                raise AttributeError("Collection not accessible")
                        except Exception as e:
                            logger.warning(f"[RETRIEVE] Direct collection access failed: {e}, using similarity search with high k")
                            # Fallback: use similarity search with very high k and generic query
                            k_well_picks = 500
                            try:
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={
                                        "k": k_well_picks,
                                        "filter": {"is_well_picks": True}
                                    }
                                )
                                well_picks_docs = well_picks_retriever.invoke("formation picks well")
                            except:
                                # Last resort: no filter, just high k
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": k_well_picks}
                                )
                                well_picks_docs = well_picks_retriever.invoke("formation picks well")
                                # Filter manually
                                well_picks_docs = [doc for doc in well_picks_docs 
                                                  if doc.metadata.get('is_well_picks') or 
                                                  'Well_picks' in doc.metadata.get('source', '') or
                                                  'Well_picks' in doc.metadata.get('filename', '')]
                            
                            if well_picks_docs:
                                # Filter by well name if specified (for comprehensive lists, this filters the full set)
                                if well_name and not is_comprehensive_list:
                                    well_picks_docs = self._filter_docs_by_well(well_picks_docs, well_name)
                                all_docs.extend(well_picks_docs)
                                logger.info(f"[RETRIEVE] Found {len(well_picks_docs)} well picks chunks via similarity search")
                    else:
                        # Regular formation query - use similarity search
                        k_well_picks = 100 if needs_multiple else 15
                        
                        # First, try to get well picks chunks using metadata filtering
                        try:
                            # Method 1: Direct metadata filter (ChromaDB native)
                            well_picks_retriever = self.vectorstore.as_retriever(
                                search_type="similarity",
                                search_kwargs={
                                    "k": k_well_picks,
                                    "filter": {"is_well_picks": {"$eq": True}}  # ChromaDB filter syntax
                                }
                            )
                            well_picks_docs = well_picks_retriever.invoke(query)
                        except:
                            # Method 2: Try boolean filter
                            try:
                                well_picks_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={
                                        "k": k_well_picks,
                                        "filter": {"is_well_picks": True}
                                    }
                                )
                                well_picks_docs = well_picks_retriever.invoke(query)
                            except:
                                # Method 3: Get all well picks chunks by searching with high k
                                well_picks_query = "formation picks well" if needs_multiple else query
                                general_retriever = self.vectorstore.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": k_well_picks * 2}
                                )
                                general_docs = general_retriever.invoke(well_picks_query)
                                well_picks_docs = [doc for doc in general_docs 
                                                  if doc.metadata.get('is_well_picks') or 
                                                  doc.metadata.get('is_well_picks') == True or
                                                  'Well_picks' in doc.metadata.get('source', '') or
                                                  'Well_picks' in doc.metadata.get('filename', '')]
                        
                        if well_picks_docs:
                            # Filter by well name if specified
                            if well_name:
                                original_count = len(well_picks_docs)
                                well_picks_docs = self._filter_docs_by_well(well_picks_docs, well_name)
                                if well_picks_docs:
                                    logger.info(f"[RETRIEVE] Filtered to {len(well_picks_docs)} chunks for well {well_name} (from {original_count})")
                                else:
                                    logger.warning(f"[RETRIEVE] No chunks found for well {well_name} after filtering, using all {original_count} chunks")
                                    # Re-fetch without filtering if filtering removed all results
                                    well_picks_retriever = self.vectorstore.as_retriever(
                                        search_type="similarity",
                                        search_kwargs={"k": k_well_picks}
                                    )
                                    well_picks_docs = well_picks_retriever.invoke(query)
                            
                            # For depth queries, get ALL chunks for the well (not just one)
                            if is_depth_query and well_name and well_picks_docs:
                                # Try to get all chunks for this well from the collection
                                try:
                                    collection = self.vectorstore._collection
                                    if collection:
                                        # Get all well picks chunks for this well
                                        all_well_chunks = []
                                        # Search for all chunks containing the well name
                                        results = collection.get(
                                            where={"is_well_picks": True},
                                            limit=10000
                                        )
                                        if results and 'documents' in results:
                                            for i, doc_text in enumerate(results['documents']):
                                                metadata = results['metadatas'][i] if 'metadatas' in results and i < len(results['metadatas']) else {}
                                                # Check if this chunk is for the queried well
                                                if self._filter_docs_by_well([Document(page_content=doc_text, metadata=metadata)], well_name):
                                                    all_well_chunks.append(Document(page_content=doc_text, metadata=metadata))
                                        if all_well_chunks:
                                            well_picks_docs = all_well_chunks
                                            logger.info(f"[RETRIEVE] Retrieved ALL {len(all_well_chunks)} chunks for well {well_name} for depth query")
                                except Exception as e:
                                    logger.warning(f"[RETRIEVE] Failed to get all chunks for well: {e}")
                            
                            all_docs.extend(well_picks_docs)
                            logger.info(f"[RETRIEVE] Found {len(well_picks_docs)} well picks chunks for formation query")
                except Exception as e:
                    logger.warning(f"[RETRIEVE] Well picks retrieval failed, using fallback: {e}")
                    # Fallback: search for well picks by filename in results
                    try:
                        # Get many documents to filter from
                        general_retriever = self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 100}
                        )
                        general_docs = general_retriever.invoke(query)
                        well_picks_docs = [doc for doc in general_docs 
                                          if doc.metadata.get('is_well_picks') or 
                                          doc.metadata.get('is_well_picks') == True or
                                          'Well_picks' in doc.metadata.get('source', '') or
                                          'Well_picks' in doc.metadata.get('filename', '')]
                        if well_picks_docs:
                            all_docs.extend(well_picks_docs)
                            logger.info(f"[RETRIEVE] Found {len(well_picks_docs)} well picks chunks via fallback")
                    except Exception as e2:
                        logger.warning(f"[RETRIEVE] Fallback also failed: {e2}")
            
            # Get general results (if not already got well picks, or need more)
            if not all_docs or needs_multiple:
                try:
                    if needs_multiple:
                        general_retriever = self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 20}
                        )
                        general_docs = general_retriever.invoke(query)
                    else:
                        general_docs = self.retriever.invoke(query)
                    
                    # Add general docs, avoiding duplicates
                    existing_sources = {doc.metadata.get('source', '') for doc in all_docs}
                    for doc in general_docs:
                        if doc.metadata.get('source', '') not in existing_sources:
                            all_docs.append(doc)
                            existing_sources.add(doc.metadata.get('source', ''))
                except Exception as e:
                    logger.warning(f"[RETRIEVE] General retrieval failed: {e}")
            
            # If still no docs, use default retriever
            if not all_docs:
                docs = self.retriever.invoke(query)
                all_docs = docs
            
            # DOCUMENT-LEVEL RETRIEVAL: Retrieve ALL chunks from relevant documents
            # Identify unique document sources from initial retrieval
            relevant_sources = set()
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                if source:
                    relevant_sources.add(source)
            
            if relevant_sources:
                logger.info(f"[RETRIEVE] Identified {len(relevant_sources)} relevant document(s), retrieving ALL chunks")
                # Retrieve all chunks from each relevant document
                full_document_chunks = self._retrieve_all_chunks_from_documents(list(relevant_sources))
                
                if full_document_chunks:
                    # Replace initial chunks with full document chunks
                    # This ensures we have complete document context, not just top-k chunks
                    all_docs = full_document_chunks
                    logger.info(f"[RETRIEVE] Retrieved ALL chunks from {len(relevant_sources)} document(s) - total {len(all_docs)} chunks")
                else:
                    logger.warning("[RETRIEVE] Full document retrieval failed, using initial retrieval results")
            
            # For comprehensive queries, we already have all chunks, so no limit needed
            # But keep the logic for backward compatibility and logging
            if is_comprehensive_list:
                logger.info("[RETRIEVE] Comprehensive list query - using ALL chunks from relevant documents")
            elif needs_multiple and is_formation_query:
                logger.info("[RETRIEVE] Multiple source query - using ALL chunks from relevant documents")
            elif needs_multiple:
                logger.info("[RETRIEVE] Multiple source query - using ALL chunks from relevant documents")
            else:
                logger.info(f"[RETRIEVE] Single query - using ALL chunks from {len(relevant_sources)} relevant document(s)")
            
            # No max_docs limit - we want ALL chunks from relevant documents
            logger.info(f"[RETRIEVE] Returning {len(all_docs)} document chunks (full documents)")
            
            # Phase 1.5: Include source and page information in output for better citation
            formatted_chunks = []
            for doc in all_docs:
                source = doc.metadata.get('source', '')
                page = doc.metadata.get('page') or doc.metadata.get('page_number')
                page_start = doc.metadata.get('page_start')
                page_end = doc.metadata.get('page_end')
                
                # Build citation string
                citation_parts = []
                if source:
                    citation_parts.append(f"Source: {source}")
                if page_start is not None and page_end is not None:
                    if page_start == page_end:
                        citation_parts.append(f"(page {page_start})")
                    else:
                        citation_parts.append(f"(pages {page_start}-{page_end})")
                elif page is not None:
                    citation_parts.append(f"(page {page})")
                
                citation = " ".join(citation_parts) if citation_parts else ""
                
                # Format chunk with citation
                if citation:
                    formatted_chunks.append(f"{doc.page_content}\n[{citation}]")
                else:
                    formatted_chunks.append(doc.page_content)
            
            return "\n\n".join(formatted_chunks)
        
        return retrieve_petrophysical_docs

