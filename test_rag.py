"""
Test script for the RAG system with complex queries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import VolveRAGSystem

def test_queries():
    """Test the RAG system with complex queries."""
    
    print("=" * 80)
    print("Testing Volve Wells RAG System")
    print("=" * 80)
    
    # Initialize system (use existing index if available)
    print("\nInitializing RAG system...")
    try:
        system = VolveRAGSystem(rebuild_index=False)
        print("✓ System initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing system: {e}")
        print("Trying to rebuild index...")
        system = VolveRAGSystem(rebuild_index=True)
        print("✓ System initialized and index rebuilt")
    
    # Test queries
    test_queries = [
        "What is the average porosity in the Hugin formation?",
        "Which well has the highest permeability?",
        "What is the porosity of well 15/9-F-1?",
        "What is the average water saturation across all wells?",
        "Compare porosity between wells in the Hugin formation",
        "What is the minimum shale volume in the Sleipner formation?",
    ]
    
    print("\n" + "=" * 80)
    print("Running Test Queries")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")
        
        try:
            result = system.query(query, n_results=20)
            
            print(f"\n📝 Answer:")
            print(result.get('answer', 'No answer generated.'))
            
            if result.get('aggregated_data'):
                agg = result['aggregated_data']
                print(f"\n📊 Aggregated Data:")
                print(f"  Value: {agg.get('value', 'N/A')}")
                print(f"  Count: {agg.get('count', 'N/A')} wells")
                if 'wells' in agg:
                    print(f"  Wells: {', '.join(agg['wells'][:5])}")
            
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for j, source in enumerate(result['sources'][:3], 1):
                    print(f"  {j}. {source.get('well_name', 'Unknown')}")
                    if source.get('formation'):
                        print(f"     Formation: {source['formation']}")
            
            print("\n✓ Query processed successfully")
            
        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_queries()

