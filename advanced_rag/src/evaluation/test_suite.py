"""
Test suite with ground truth queries for RAG evaluation.
"""

from typing import List, Dict, Any

# Sample test queries with expected relevant documents
# In a real scenario, these would be manually curated with ground truth
TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "What formations are in well 15/9-F-5?",
        "relevant_docs": ["Well_picks_Volve_v1.dat"],
        "category": "formation_listing"
    },
    {
        "query": "What is the porosity for Hugin formation in well 15/9-F-5?",
        "relevant_docs": ["PETROPHYSICAL_REPORT_1.PDF"],  # Example - should be actual doc IDs
        "category": "petrophysical_property"
    },
    {
        "query": "What is the depth of Sleipner formation in well 15/9-19A?",
        "relevant_docs": ["Well_picks_Volve_v1.dat"],
        "category": "depth_query"
    },
    {
        "query": "What is the fluid density for Hugin in 15/9-F-5?",
        "relevant_docs": ["PETROPHYSICAL_REPORT_1.PDF"],  # Example
        "category": "evaluation_parameter"
    },
    {
        "query": "list all formations and their properties",
        "relevant_docs": ["Well_picks_Volve_v1.dat", "PETROPHYSICAL_REPORT_1.PDF"],
        "category": "comprehensive_listing"
    },
    {
        "query": "What is the Archie n parameter for Hugin in 15/9-F-5?",
        "relevant_docs": ["PETROPHYSICAL_REPORT_1.PDF"],
        "category": "evaluation_parameter"
    },
    {
        "query": "What is the net to gross for Sleipner in 15/9-F-5?",
        "relevant_docs": ["PETROPHYSICAL_REPORT_1.PDF"],
        "category": "petrophysical_property"
    },
    {
        "query": "formations in Well NO 15/9-F-15 A",
        "relevant_docs": ["Well_picks_Volve_v1.dat"],
        "category": "formation_listing"
    },
    {
        "query": "What is the permeability for Hugin formation in 15/9-F-5?",
        "relevant_docs": ["PETROPHYSICAL_REPORT_1.PDF"],
        "category": "petrophysical_property"
    },
    {
        "query": "What is the matrix density for Hugin in 15/9-F-5?",
        "relevant_docs": ["PETROPHYSICAL_REPORT_1.PDF"],
        "category": "evaluation_parameter"
    },
]


def get_test_queries() -> List[Dict[str, Any]]:
    """Get the test query suite."""
    return TEST_QUERIES


def get_queries_by_category(category: str) -> List[Dict[str, Any]]:
    """Get test queries filtered by category."""
    return [q for q in TEST_QUERIES if q.get("category") == category]

