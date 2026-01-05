"""
Compare hybrid fusion methods (weighted merge vs Reciprocal Rank Fusion).

This script is intentionally lightweight and meant for local verification/debugging.
It disables MMR, cross-encoder reranking, and LLM reranking so you can see the
effect of the fusion method itself.

Usage:
    python scripts/compare_fusion.py
    python scripts/compare_fusion.py --k-final 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

# Add project root (advanced_rag/) to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.tools.retriever_tool import RetrieverTool


DEFAULT_QUERIES = [
    ("formation pressure characteristics Hugin 15/9-F-1 C", "PETROPHYSICAL_REPORT_1.PDF"),
    ("15/9-F-1 C perforations", "PETROPHYSICAL_REPORT_1.PDF"),
    ("Baker’s TesTrak Formation Pressure While Drilling tool 15/9-F-1 C", "PETROPHYSICAL_REPORT_1.PDF"),
    ("PRES_QUAL formation pressure test quality OpenWorks 15/9-F-1 C", "PETROPHYSICAL_REPORT_1.PDF"),
    ("pressure level change offshore shale Hugin 3.2 15/9-F-1 C", "PETROPHYSICAL_REPORT_1.PDF"),
]


def _summarize_docs(docs, k_final: int):
    out = []
    for d in docs[:k_final]:
        src = d.metadata.get("source", "") or ""
        name = Path(str(src)).name if src else ""
        page = d.metadata.get("page", d.metadata.get("page_number", ""))
        out.append((name, page, src))
    return out


def _run_once(retriever: RetrieverTool, fusion: str, query: str, k_final: int):
    retriever._fusion_method = fusion
    t0 = time.time()
    docs = retriever._hybrid_retrieve([query], k_vec=20, k_lex=30, k_final=max(k_final, 10))
    # Apply the same well filtering used in the production retrieval tool wrapper
    try:
        well_name = retriever._extract_well_name(query)
    except Exception:
        well_name = None
    if well_name:
        try:
            docs = retriever._filter_docs_by_well(docs, well_name)
        except Exception:
            pass
    dt = time.time() - t0
    return dt, _summarize_docs(docs, k_final=k_final)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare weighted hybrid merge vs RRF fusion.")
    parser.add_argument("--vectorstore", type=str, default="./data/vectorstore", help="Path to vectorstore")
    parser.add_argument("--k-final", type=int, default=8, help="Number of results to show per fusion method")
    args = parser.parse_args()

    r = RetrieverTool(persist_directory=args.vectorstore)
    if not r.load_vectorstore():
        print("ERROR: Vector store not found. Run index build first.")
        return 1

    # Isolate fusion effect (avoid MMR/rerank changing ordering)
    r._mmr_enabled = False
    r._use_cross_encoder = False
    r._rerank_enabled = False

    for query, expected_name in DEFAULT_QUERIES:
        print("=" * 90)
        print(f"QUERY: {query}")
        print(f"Expected file: {expected_name}")

        for fusion in ("weighted", "rrf"):
            dt, items = _run_once(r, fusion=fusion, query=query, k_final=args.k_final)
            print(f"\n[{fusion}] {dt:.2f}s")
            for i, (name, page, src) in enumerate(items, start=1):
                print(f"  {i:>2}. {name}  p.{page}  ({src})")
            hit = any(name == expected_name for name, _page, _src in items)
            print(f"  HIT_EXPECTED: {hit}")

    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


