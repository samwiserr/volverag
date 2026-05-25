"""
SOTA RAG Evaluation using RAGAS.

Evaluates the full VolveRAG pipeline on the curated Volve test suite using
four RAGAS metrics:

  - Faithfulness       : Are the answer claims grounded in the retrieved context?
  - Answer Relevancy   : Is the answer relevant to the question?
  - Context Precision  : Are retrieved chunks actually useful for the answer?
  - Context Recall     : Were all ground-truth facts retrieved?

Usage:
    cd advanced_rag
    python scripts/sota_evaluation.py [--output results/sota_eval.json]

Requires:
    pip install -r requirements-eval.txt   # ragas + datasets (local only)
    OPENAI_API_KEY set in environment (or .env file)

The script loads the existing vector store (must have been built with
`python -m src.main --build-index`) and runs the full query → retrieve →
generate pipeline for each test query, then calls RAGAS metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Bootstrap path so imports work when running from scripts/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load .env before any other import
from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth answer snippets (hand-labelled from actual Volve reports)
# Used only for Context Recall evaluation.
# ---------------------------------------------------------------------------
GROUND_TRUTH: Dict[str, str] = {
    "What formations are in well 15/9-F-5?":
        "Hugin, Sleipner, Skagerrak, Smith Bank, Zechstein formations are present in well 15/9-F-5.",
    "What is the porosity for Hugin formation in well 15/9-F-5?":
        "The effective porosity (PHIF) for Hugin formation in 15/9-F-5 is approximately 0.22–0.27.",
    "What is the depth of Sleipner formation in well 15/9-19A?":
        "Sleipner formation top in well 15/9-19A has a measured depth (MD) around 2730 m.",
    "What is the fluid density for Hugin in 15/9-F-5?":
        "The fluid density (RhoFl) used for Hugin in 15/9-F-5 is 0.65 g/cc (gas).",
    "list all formations and their properties":
        "Volve wells penetrate Hugin, Sleipner, Skagerrak, Smith Bank, and Zechstein formations with varying porosity, permeability, and saturation.",
    "What is the Archie n parameter for Hugin in 15/9-F-5?":
        "The Archie saturation exponent n for Hugin in 15/9-F-5 is 2.0.",
    "What is the net to gross for Sleipner in 15/9-F-5?":
        "Net-to-gross for Sleipner formation in 15/9-F-5 is approximately 0.70–0.80.",
    "formations in Well NO 15/9-F-15 A":
        "Well NO 15/9-F-15A penetrates Hugin, Sleipner, and Skagerrak formations.",
    "What is the permeability for Hugin formation in 15/9-F-5?":
        "Permeability (KLOGH) for Hugin formation in 15/9-F-5 ranges from 100 to 2000 mD.",
    "What is the matrix density for Hugin in 15/9-F-5?":
        "The matrix density (RhoMa) for Hugin in 15/9-F-5 is 2.65 g/cc (quartz-dominated sandstone).",
}


def _load_rag_pipeline():
    """Load the RetrieverTool + LangGraph graph from the built vector store."""
    from src.tools.retriever_tool import RetrieverTool
    from src.graph.rag_graph import build_rag_graph
    from src.tools.well_picks_tool import WellPicksTool
    from src.tools.petro_params_tool import PetroParamsTool

    rt = RetrieverTool()
    if not rt.load_vectorstore():
        raise RuntimeError(
            "Vector store not found. Run: python -m src.main --build-index first."
        )

    tools = [rt.get_retriever_tool()]
    # Optionally add structured tools if available
    try:
        wp = WellPicksTool()
        tools.append(wp.get_tool())
    except Exception:
        pass
    try:
        pp = PetroParamsTool()
        tools.append(pp.get_tool())
    except Exception:
        pass

    graph = build_rag_graph(tools)
    return graph, rt


def _run_query(graph, query: str) -> Dict[str, Any]:
    """Run a single query through the full RAG graph and return answer + contexts."""
    from langchain_core.messages import HumanMessage
    start = time.perf_counter()
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    elapsed = time.perf_counter() - start

    messages = result.get("messages", [])
    answer = ""
    contexts: List[str] = []
    for msg in messages:
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            answer = msg.content
        if hasattr(msg, "content") and msg.__class__.__name__ == "ToolMessage":
            contexts.append(str(msg.content)[:3000])
    return {"answer": answer, "contexts": contexts, "latency_s": elapsed}


def _run_ragas(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run RAGAS evaluation on a list of samples."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError as exc:
        logger.error(f"RAGAS not installed: {exc}")
        return {}

    data = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }
    ds = Dataset.from_dict(data)
    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }


def main():
    parser = argparse.ArgumentParser(description="SOTA RAGAS evaluation for VolveRAG")
    parser.add_argument(
        "--output",
        default="results/sota_eval.json",
        help="Path to write JSON results (default: results/sota_eval.json)",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        help="Subset of query keys to evaluate (default: all 10)",
    )
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)

    print("\n=== VolveRAG SOTA Evaluation (RAGAS) ===\n")
    print("Loading pipeline…")
    graph, _ = _load_rag_pipeline()

    from src.evaluation.test_suite import get_test_queries
    test_qs = get_test_queries()
    if args.queries:
        test_qs = [q for q in test_qs if q["query"] in args.queries]

    samples: List[Dict[str, Any]] = []
    per_query_results: List[Dict[str, Any]] = []

    for i, tq in enumerate(test_qs, 1):
        query = tq["query"]
        gt = GROUND_TRUTH.get(query, "")
        print(f"[{i}/{len(test_qs)}] {query[:80]}…")
        try:
            resp = _run_query(graph, query)
            print(f"  Answer ({resp['latency_s']:.1f}s): {resp['answer'][:120]}…")
            samples.append(
                {
                    "question": query,
                    "answer": resp["answer"],
                    "contexts": resp["contexts"] or ["(no context retrieved)"],
                    "ground_truth": gt,
                }
            )
            per_query_results.append(
                {
                    "query": query,
                    "category": tq.get("category"),
                    "answer": resp["answer"],
                    "latency_s": resp["latency_s"],
                    "n_contexts": len(resp["contexts"]),
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            per_query_results.append({"query": query, "error": str(exc)})

    print("\nRunning RAGAS metrics…")
    ragas_scores = _run_ragas(samples)

    if ragas_scores:
        print("\n── RAGAS Results ──────────────────────────────")
        for k, v in ragas_scores.items():
            print(f"  {k:<22}: {v:.4f}")
        print("────────────────────────────────────────────────\n")
    else:
        print("RAGAS scoring skipped (install ragas and datasets packages)\n")

    avg_latency = sum(r.get("latency_s", 0) for r in per_query_results) / max(len(per_query_results), 1)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_queries": len(test_qs),
        "avg_latency_s": round(avg_latency, 2),
        "ragas_scores": ragas_scores,
        "per_query": per_query_results,
        "sota_techniques": [
            "ContextualChunker (Anthropic Contextual Retrieval, Sept 2024)",
            "HyDE — Hypothetical Document Embeddings (Gao et al., SIGIR 2023)",
            "RAPTOR hierarchical summarization (Sarthi et al., ICLR 2024)",
            "Hybrid retrieval: BM25 + local dense embeddings (nomic-ai/nomic-embed-text-v1.5) with RRF fusion",
            "Cross-encoder reranking + LLM reranking pipeline",
            "MMR diversification for result diversity",
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
