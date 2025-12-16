"""
Test script to verify fixes for RAG query errors and formation retrieval.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import VolveRAGSystem

def test_formation_listing():
    """Test that listing all formations works and returns formations from all wells."""
    print("=" * 80)
    print("Testing Formation Listing Fixes")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing RAG system...")
    try:
        system = VolveRAGSystem(rebuild_index=False)
        print("   ✓ System initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test queries
    test_queries = [
        "list all formations in all wells",
        "list all available surfaces",
        "what formations are in all wells"
    ]
    
    all_passed = True
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        try:
            result = system.query(query, n_results=20)
            
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            
            print(f"   ✓ Query processed successfully")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Sources found: {len(sources)}")
            
            # Check if answer mentions multiple wells
            well_names_in_answer = set()
            for source in sources:
                well_name = source.get('well_name', '')
                if well_name:
                    well_names_in_answer.add(well_name)
            
            print(f"   Wells in sources: {len(well_names_in_answer)}")
            if well_names_in_answer:
                print(f"   Sample wells: {', '.join(list(well_names_in_answer)[:3])}")
            
            # Check for header row in answer
            if 'name Surface name Obs# Qlf MD TVD TVDSS TWT Dip Azi Easting Northing Intrp' in answer:
                print(f"   ✗ WARNING: Header row found in answer!")
                all_passed = False
            else:
                print(f"   ✓ No header rows in answer")
            
            # Check if we got formations from multiple wells
            if len(well_names_in_answer) < 2:
                print(f"   ⚠ WARNING: Only {len(well_names_in_answer)} well(s) found. Expected multiple wells.")
            else:
                print(f"   ✓ Multiple wells found")
            
            # Show first 200 chars of answer
            print(f"   Answer preview: {answer[:200]}...")
            
        except Exception as e:
            print(f"   ✗ Error processing query: {type(e).__name__}: {str(e)}")
            if "0" in str(e) and len(str(e)) <= 3:
                print(f"   ✗ CRITICAL: Still getting cryptic '0' error!")
                all_passed = False
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed or had warnings")
    print("=" * 80)
    
    return all_passed

def test_error_handling():
    """Test that error handling is improved and no cryptic '0' errors occur."""
    print("\n" + "=" * 80)
    print("Testing Error Handling")
    print("=" * 80)
    
    # Initialize system
    print("\n1. Initializing RAG system...")
    try:
        system = VolveRAGSystem(rebuild_index=False)
        print("   ✓ System initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize system: {e}")
        return False
    
    # Test with various queries that might cause errors
    test_queries = [
        "list all formations in all wells",
        "what is the average porosity",
        "show me formations",
    ]
    
    errors_found = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        try:
            result = system.query(query, n_results=20)
            print(f"   ✓ Query processed successfully")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"   ✗ Error: {error_type}: {error_msg}")
            
            # Check for cryptic '0' error
            if error_msg == "0" or (len(error_msg) <= 3 and "0" in error_msg):
                errors_found.append(f"Query '{query}': Cryptic '0' error")
                print(f"   ✗ CRITICAL: Cryptic '0' error detected!")
            else:
                print(f"   ✓ Error message is informative")
    
    print("\n" + "=" * 80)
    if errors_found:
        print(f"✗ Found {len(errors_found)} cryptic errors:")
        for err in errors_found:
            print(f"  - {err}")
        return False
    else:
        print("✓ No cryptic errors found")
        return True

if __name__ == "__main__":
    print("Testing RAG Query Fixes")
    print("=" * 80)
    
    # Test error handling first
    error_test_passed = test_error_handling()
    
    # Test formation listing
    formation_test_passed = test_formation_listing()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Error Handling Test: {'✓ PASSED' if error_test_passed else '✗ FAILED'}")
    print(f"Formation Listing Test: {'✓ PASSED' if formation_test_passed else '✗ FAILED'}")
    
    if error_test_passed and formation_test_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)

