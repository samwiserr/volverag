"""
CLI script to run RAG system evaluation.

Usage:
    python scripts/run_evaluation.py [--baseline] [--compare] [--export PATH]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.test_suite import get_test_queries
from src.tools.retriever_tool import RetrieverTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_retriever_func(retriever_tool: RetrieverTool):
    """Create a function that retrieves documents for a query."""
    def retrieve(query: str):
        """Retrieve documents for a query."""
        try:
            # Use the retriever tool's hybrid retrieval
            docs = retriever_tool._hybrid_retrieve([query], k_vec=20, k_lex=30, k_final=10)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving for query '{query}': {e}")
            return []
    
    return retrieve


def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--vectorstore", type=str, default="./data/vectorstore", help="Path to vectorstore")
    
    args = parser.parse_args()
    
    # Initialize retriever tool
    logger.info(f"[EVAL] Loading vectorstore from {args.vectorstore}")
    retriever_tool = RetrieverTool(persist_directory=args.vectorstore)
    retriever_tool.load_vectorstore()
    
    # Create retriever function
    retriever_func = create_retriever_func(retriever_tool)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(retriever_func)
    
    # Run evaluation
    export_path = args.export
    if args.baseline:
        export_path = export_path or "./data/evaluation/baseline_results.json"
    
    results = runner.run_evaluation(export_path=export_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of queries: {results['num_queries']}")
    print(f"\nAverage Metrics:")
    for metric, value in results['average_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Compare with baseline if requested
    if args.compare:
        baseline_path = Path("./data/evaluation/baseline_results.json")
        if baseline_path.exists():
            import json
            with open(baseline_path, 'r') as f:
                baseline_results = json.load(f)
            
            comparison = runner.compare_results(baseline_results, results)
            
            print("\n" + "="*60)
            print("COMPARISON WITH BASELINE")
            print("="*60)
            for metric, improvement in comparison['improvements'].items():
                abs_imp = improvement['absolute']
                pct_imp = improvement['percentage']
                sign = "+" if abs_imp >= 0 else ""
                print(f"  {metric}: {sign}{abs_imp:.4f} ({sign}{pct_imp:.2f}%)")
        else:
            logger.warning("[EVAL] Baseline results not found. Run with --baseline first.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()




