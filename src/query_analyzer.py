"""
Query analysis module for understanding user queries and determining what type of answer is needed.
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum


class QueryType(Enum):
    """Types of queries the system can handle."""
    AGGREGATION = "aggregation"  # e.g., "average porosity", "mean permeability"
    COMPARISON = "comparison"    # e.g., "compare wells", "which well has highest"
    FILTER = "filter"            # e.g., "wells with porosity > 0.2"
    SPECIFIC = "specific"        # e.g., "what is the porosity of well X"
    FORMATION_QUERY = "formation"  # e.g., "porosity in Hugin formation"
    LIST = "list"                # e.g., "list all wells", "what logs are available"
    GENERAL = "general"          # General information query


class QueryAnalyzer:
    """Analyzes user queries to determine intent and extract parameters."""
    
    # Aggregation keywords
    AGGREGATION_KEYWORDS = {
        'average', 'avg', 'mean', 'median', 'minimum', 'min', 'maximum', 'max',
        'total', 'sum', 'count', 'how many', 'what is the', 'what are the'
    }
    
    # Comparison keywords
    COMPARISON_KEYWORDS = {
        'compare', 'comparison', 'versus', 'vs', 'difference', 'better', 'best', 'worst',
        'highest', 'lowest', 'largest', 'smallest', 'top', 'bottom'
    }
    
    # List keywords
    LIST_KEYWORDS = {
        'list', 'show', 'what', 'which', 'available', 'all', 'what are', 'what is',
        'tell me', 'give me', 'show me'
    }
    
    # Analytical keywords (require computation/tools)
    ANALYTICAL_KEYWORDS = {
        'average', 'avg', 'mean', 'median', 'minimum', 'min', 'maximum', 'max',
        'total', 'sum', 'count', 'between', 'plot', 'graph', 'chart', 'compare',
        'calculate', 'compute', 'statistics', 'stats'
    }
    
    # Document-style keywords (can be answered from text)
    DOCUMENT_KEYWORDS = {
        'what logs', 'available logs', 'curves', 'summarize', 'describe', 'characteristics',
        'information', 'details', 'metadata', 'header', 'location', 'operator'
    }
    
    # Formation keywords
    FORMATION_KEYWORDS = {
        'formation', 'fm', 'fm.', 'group', 'gp', 'gp.', 'in the', 'within'
    }
    
    # Curve names and synonyms
    CURVE_SYNONYMS = {
        'porosity': ['phif', 'porosity', 'phi'],
        'permeability': ['klogh', 'permeability', 'perm', 'k'],
        'water saturation': ['sw', 'water saturation', 'saturation'],
        'shale volume': ['vsh', 'shale volume', 'shale', 'vshale'],
        'bound volume water': ['bvw', 'bound volume water']
    }
    
    def __init__(self):
        """Initialize the query analyzer."""
        pass
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze a query to determine its type and extract parameters.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with analysis results
        """
        query_lower = query.lower()
        
        analysis = {
            'query_type': QueryType.GENERAL,
            'curve': None,
            'formation': None,
            'aggregation_type': None,
            'well_name': None,
            'comparison_type': None,
            'filters': {},
            'is_analytical': False,  # Whether this requires computation/tools
            'is_document_style': False,  # Whether this can be answered from text
            'should_use_listing_tool': False  # Whether to use listing tools for complete results
        }
        
        # Check for list queries
        is_list = any(keyword in query_lower for keyword in self.LIST_KEYWORDS) and \
                 ('list' in query_lower or 'available' in query_lower or 'what' in query_lower)
        
        # Check for comparison queries FIRST (before aggregation)
        is_comparison = any(keyword in query_lower for keyword in self.COMPARISON_KEYWORDS)
        
        # Check for aggregation queries
        is_aggregation = any(keyword in query_lower for keyword in self.AGGREGATION_KEYWORDS)
        
        # Classify as analytical vs document-style
        is_analytical = any(keyword in query_lower for keyword in self.ANALYTICAL_KEYWORDS)
        is_document_style = any(keyword in query_lower for keyword in self.DOCUMENT_KEYWORDS)
        
        analysis['is_analytical'] = is_analytical
        analysis['is_document_style'] = is_document_style and not is_analytical
        
        if is_list and not is_analytical:
            analysis['query_type'] = QueryType.LIST
            # Check if this is a "list all" query that should use listing tools
            if 'all' in query_lower and ('formation' in query_lower or 'surface' in query_lower or 'well' in query_lower):
                analysis['should_use_listing_tool'] = True
        elif is_comparison:
            analysis['query_type'] = QueryType.COMPARISON
            analysis['comparison_type'] = self._extract_comparison_type(query_lower)
        elif is_aggregation:
            analysis['query_type'] = QueryType.AGGREGATION
            analysis['aggregation_type'] = self._extract_aggregation_type(query_lower)
        
        # Check for formation-specific queries
        formation = self._extract_formation(query_lower)
        if formation:
            analysis['formation'] = formation
            if analysis['query_type'] == QueryType.GENERAL:
                analysis['query_type'] = QueryType.FORMATION_QUERY
            # Don't override comparison or aggregation types
        
        # Extract curve name
        curve = self._extract_curve(query_lower)
        if curve:
            analysis['curve'] = curve
        
        # Check for comparison queries (override aggregation if both present)
        if any(keyword in query_lower for keyword in self.COMPARISON_KEYWORDS):
            analysis['query_type'] = QueryType.COMPARISON
            analysis['comparison_type'] = self._extract_comparison_type(query_lower)
            # If no curve extracted yet, try to infer from comparison
            if not analysis.get('curve'):
                if 'permeability' in query_lower or 'perm' in query_lower:
                    analysis['curve'] = 'permeability'
                elif 'porosity' in query_lower:
                    analysis['curve'] = 'porosity'
                elif 'saturation' in query_lower:
                    analysis['curve'] = 'water saturation'
                elif 'shale' in query_lower:
                    analysis['curve'] = 'shale volume'
        
        # Extract well name
        well_name = self._extract_well_name(query_lower)
        if well_name:
            analysis['well_name'] = well_name
            if analysis['query_type'] == QueryType.GENERAL:
                analysis['query_type'] = QueryType.SPECIFIC
        
        # Extract filters
        filters = self._extract_filters(query_lower)
        if filters:
            analysis['filters'] = filters
            if analysis['query_type'] == QueryType.GENERAL:
                analysis['query_type'] = QueryType.FILTER
        
        return analysis
    
    def _extract_aggregation_type(self, query: str) -> Optional[str]:
        """Extract the type of aggregation requested."""
        if 'average' in query or 'avg' in query or 'mean' in query:
            return 'mean'
        elif 'median' in query:
            return 'median'
        elif 'minimum' in query or 'min' in query:
            return 'min'
        elif 'maximum' in query or 'max' in query:
            return 'max'
        elif 'total' in query or 'sum' in query:
            return 'sum'
        elif 'count' in query or 'how many' in query:
            return 'count'
        return 'mean'  # Default to mean
    
    def _extract_formation(self, query: str) -> Optional[str]:
        """Extract formation name from query."""
        # Common formations in Volve
        formations = [
            'hugin', 'sleipner', 'skagerrak', 'smith bank', 'heather', 'draupne',
            'hod', 'ekofisk', 'ty', 'utsira', 'nordland', 'hordaland', 'shetland'
        ]
        
        for formation in formations:
            if formation in query:
                return formation.title()
        
        # Try to extract formation after keywords
        pattern = r'(?:formation|fm\.?|in the|within)\s+([A-Za-z\s]+?)(?:\s+formation|$)'
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip().title()
        
        return None
    
    def _extract_curve(self, query: str) -> Optional[str]:
        """Extract curve name from query."""
        for standard_name, synonyms in self.CURVE_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in query:
                    return standard_name
        return None
    
    def _extract_well_name(self, query: str) -> Optional[str]:
        """Extract well name from query."""
        # Pattern for well names like "15/9-F-1" or "15/9-19 A"
        pattern = r'\b(\d+[/-]\d+[-/][A-Z0-9\s]+)\b'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_comparison_type(self, query: str) -> Optional[str]:
        """Extract comparison type."""
        if 'highest' in query or 'largest' in query or 'best' in query or 'top' in query:
            return 'max'
        elif 'lowest' in query or 'smallest' in query or 'worst' in query or 'bottom' in query:
            return 'min'
        return None
    
    def _extract_filters(self, query: str) -> Dict:
        """Extract filter conditions from query."""
        filters = {}
        
        # Numeric comparisons
        patterns = {
            'phif': r'porosity\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
            'klogh': r'permeability\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
            'sw': r'(?:water\s+)?saturation\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
        }
        
        for curve_key, pattern in patterns.items():
            match = re.search(pattern, query)
            if match:
                operator = match.group(1).replace('=', '==')
                value = float(match.group(2))
                filters[curve_key] = {'operator': operator, 'value': value}
        
        return filters


if __name__ == "__main__":
    # Test the query analyzer
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "What is the average porosity in the Hugin formation?",
        "Which well has the highest permeability?",
        "Compare porosity between wells in the Hugin formation",
        "What is the porosity of well 15/9-F-1?",
        "Find wells with porosity > 0.2"
    ]
    
    for query in test_queries:
        analysis = analyzer.analyze(query)
        print(f"\nQuery: {query}")
        print(f"Type: {analysis['query_type'].value}")
        print(f"Curve: {analysis['curve']}")
        print(f"Formation: {analysis['formation']}")
        print(f"Aggregation: {analysis['aggregation_type']}")

