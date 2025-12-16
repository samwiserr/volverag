"""
Aggregation module for calculating statistics across multiple wells or formations.
"""

from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


class DataAggregator:
    """Aggregates statistics across multiple wells or formations."""
    
    def __init__(self):
        """Initialize the aggregator."""
        pass
    
    def aggregate_by_formation(self, results: List[Dict], formation: str, 
                               curve: str, aggregation_type: str = 'mean') -> Optional[Dict]:
        """
        Aggregate statistics for a specific formation.
        
        Args:
            results: List of search results
            formation: Formation name
            curve: Curve name (e.g., 'porosity', 'permeability')
            aggregation_type: Type of aggregation ('mean', 'median', 'min', 'max', etc.)
            
        Returns:
            Dictionary with aggregated statistics
        """
        # Map curve names to metadata keys
        curve_map = {
            'porosity': 'PHIF',
            'permeability': 'KLOGH',
            'water saturation': 'SW',
            'shale volume': 'VSH',
            'bound volume water': 'BVW'
        }
        
        curve_key = curve_map.get(curve.lower(), curve.upper())
        metadata_key = f'{curve_key}_mean'
        
        # Filter results by formation
        formation_results = []
        for result in results:
            metadata = result.get('metadata', {})
            result_formation = metadata.get('formation_name', '')
            
            # Check if formation matches (case-insensitive, partial match)
            if formation.lower() in result_formation.lower() or result_formation.lower() in formation.lower():
                # Check if this result has the curve data
                if metadata_key in metadata and metadata[metadata_key] is not None:
                    formation_results.append(result)
        
        # Also check if formation is mentioned in the document
        if not formation_results:
            for result in results:
                doc = result.get('document', '').lower()
                if formation.lower() in doc:
                    metadata = result.get('metadata', {})
                    if metadata_key in metadata and metadata[metadata_key] is not None:
                        formation_results.append(result)
        
        if not formation_results:
            return None
        
        # Extract values
        values = []
        well_names = []
        for result in formation_results:
            metadata = result.get('metadata', {})
            value = metadata.get(metadata_key)
            if value is not None:
                values.append(float(value))
                well_names.append(metadata.get('well_name', 'Unknown'))
        
        if not values:
            return None
        
        # Calculate aggregation
        aggregated = self._calculate_aggregation(values, aggregation_type)
        
        return {
            'value': aggregated,
            'count': len(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'wells': list(set(well_names)),
            'formation': formation,
            'curve': curve
        }
    
    def aggregate_all(self, results: List[Dict], curve: str, 
                     aggregation_type: str = 'mean') -> Optional[Dict]:
        """
        Aggregate statistics across all results.
        
        Args:
            results: List of search results
            curve: Curve name
            aggregation_type: Type of aggregation
            
        Returns:
            Dictionary with aggregated statistics
        """
        curve_map = {
            'porosity': 'PHIF',
            'permeability': 'KLOGH',
            'water saturation': 'SW',
            'shale volume': 'VSH',
            'bound volume water': 'BVW'
        }
        
        curve_key = curve_map.get(curve.lower(), curve.upper())
        metadata_key = f'{curve_key}_mean'
        
        # Extract values
        values = []
        well_names = []
        for result in results:
            metadata = result.get('metadata', {})
            value = metadata.get(metadata_key)
            if value is not None:
                values.append(float(value))
                well_names.append(metadata.get('well_name', 'Unknown'))
        
        if not values:
            return None
        
        # Calculate aggregation
        aggregated = self._calculate_aggregation(values, aggregation_type)
        
        return {
            'value': aggregated,
            'count': len(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'wells': list(set(well_names)),
            'curve': curve
        }
    
    def compare_wells(self, results: List[Dict], curve: str, 
                     comparison_type: str = 'max') -> Optional[Dict]:
        """
        Compare wells and find the one with highest/lowest value.
        
        Args:
            results: List of search results
            curve: Curve name
            comparison_type: 'max' or 'min'
            
        Returns:
            Dictionary with comparison results
        """
        curve_map = {
            'porosity': 'PHIF',
            'permeability': 'KLOGH',
            'water saturation': 'SW',
            'shale volume': 'VSH'
        }
        
        curve_key = curve_map.get(curve.lower(), curve.upper())
        metadata_key = f'{curve_key}_mean'
        
        # Extract well-value pairs
        well_values = {}
        for result in results:
            metadata = result.get('metadata', {})
            well_name = metadata.get('well_name')
            value = metadata.get(metadata_key)
            
            if well_name and value is not None:
                if well_name not in well_values or (
                    (comparison_type == 'max' and value > well_values[well_name]) or
                    (comparison_type == 'min' and value < well_values[well_name])
                ):
                    well_values[well_name] = float(value)
        
        if not well_values:
            return None
        
        # Find the well with max/min value
        if comparison_type == 'max':
            best_well = max(well_values, key=well_values.get)
            best_value = well_values[best_well]
        else:
            best_well = min(well_values, key=well_values.get)
            best_value = well_values[best_well]
        
        return {
            'well_name': best_well,
            'value': best_value,
            'curve': curve,
            'comparison_type': comparison_type,
            'all_values': well_values
        }
    
    def _calculate_aggregation(self, values: List[float], aggregation_type: str) -> float:
        """Calculate the specified aggregation."""
        if not values:
            return 0.0
        
        if aggregation_type == 'mean' or aggregation_type == 'average' or aggregation_type == 'avg':
            return float(np.mean(values))
        elif aggregation_type == 'median':
            return float(np.median(values))
        elif aggregation_type == 'min' or aggregation_type == 'minimum':
            return float(np.min(values))
        elif aggregation_type == 'max' or aggregation_type == 'maximum':
            return float(np.max(values))
        elif aggregation_type == 'sum' or aggregation_type == 'total':
            return float(np.sum(values))
        elif aggregation_type == 'count':
            return float(len(values))
        else:
            return float(np.mean(values))  # Default to mean
    
    def filter_by_formation_depth(self, results: List[Dict], formation: str, 
                                 formation_tops_data: Dict) -> List[Dict]:
        """
        Filter results to only include data within formation depth intervals.
        
        Args:
            results: List of search results
            formation: Formation name
            formation_tops_data: Dictionary mapping well names to formation tops
            
        Returns:
            Filtered list of results
        """
        filtered = []
        
        for result in results:
            metadata = result.get('metadata', {})
            well_name = metadata.get('well_name', '')
            start_depth = metadata.get('start_depth')
            end_depth = metadata.get('end_depth')
            
            # Find formation top for this well
            formation_top = None
            formation_base = None
            
            for key, formations in formation_tops_data.items():
                if well_name.upper() in key.upper() or key.upper() in well_name.upper():
                    for ft in formations:
                        if formation.lower() in ft.get('formation_name', '').lower():
                            formation_top = ft.get('md')
                            # Look for base (next formation or Hugin Base)
                            break
                    # Also check for "Hugin Fm. VOLVE Base" or similar
                    for ft in formations:
                        if 'base' in ft.get('formation_name', '').lower() and formation.lower() in ft.get('formation_name', '').lower():
                            formation_base = ft.get('md')
                            break
                    break
            
            # If we have formation depth info, check if result overlaps
            if formation_top:
                if start_depth and end_depth:
                    # Check if interval overlaps with formation
                    if (start_depth <= formation_base if formation_base else True) and end_depth >= formation_top:
                        filtered.append(result)
                else:
                    # No depth info, include it
                    filtered.append(result)
            else:
                # No formation top found, include if document mentions formation
                doc = result.get('document', '').lower()
                if formation.lower() in doc:
                    filtered.append(result)
        
        return filtered if filtered else results  # Return original if no matches


if __name__ == "__main__":
    # Test the aggregator
    aggregator = DataAggregator()
    
    test_results = [
        {
            'metadata': {'well_name': '15/9-F-1', 'PHIF_mean': 0.15, 'formation_name': 'Hugin'},
            'document': 'Well with porosity 0.15'
        },
        {
            'metadata': {'well_name': '15/9-F-1A', 'PHIF_mean': 0.18, 'formation_name': 'Hugin'},
            'document': 'Well with porosity 0.18'
        },
        {
            'metadata': {'well_name': '15/9-19A', 'PHIF_mean': 0.16, 'formation_name': 'Hugin'},
            'document': 'Well with porosity 0.16'
        }
    ]
    
    aggregated = aggregator.aggregate_by_formation(
        test_results, 'Hugin', 'porosity', 'mean'
    )
    
    print("Aggregated Result:")
    print(aggregated)

