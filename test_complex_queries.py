"""
Test complex reservoir and formation queries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import VolveRAGSystem

def main():
    print("=" * 80)
    print("Testing Complex Reservoir and Formation Queries")
    print("=" * 80)
    
    # Initialize system
    print("\nInitializing RAG system...")
    system = VolveRAGSystem(rebuild_index=False)
    print("System ready!\n")
    
    # Complex test queries
    test_queries = [
        "What is the average porosity in the Hugin formation?",
        "Which well has the highest permeability?",
        "What is the porosity of well 15/9-F-1?",
        "What is the average water saturation across all wells?",
        "What is the minimum shale volume in the Sleipner formation?",
        "Compare porosity between wells in the Hugin formation",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")
        
        try:
            result = system.query(query, n_results=20)
            
            print(f"\nAnswer:")
            print(result['answer'])
            
            if result.get('aggregated_data'):
                agg = result['aggregated_data']
                print(f"\nAggregated Statistics:")
                if 'value' in agg:
                    print(f"  Value: {agg['value']:.4f}")
                if 'count' in agg:
                    print(f"  Based on: {agg['count']} well(s)")
                if 'wells' in agg and agg['wells']:
                    print(f"  Wells: {', '.join(agg['wells'][:5])}")
            
            if result.get('sources'):
                print(f"\nSources ({len(result['sources'])}):")
                for j, source in enumerate(result['sources'][:3], 1):
                    print(f"  {j}. {source.get('well_name', 'Unknown')}")
                    if source.get('formation'):
                        print(f"     Formation: {source['formation']}")
            
            print("\n[SUCCESS]")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Testing Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()

