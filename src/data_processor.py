"""
Data processing module for structuring well data, calculating statistics, and creating summaries.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class WellDataProcessor:
    """Processes well data to create structured summaries and statistics."""
    
    # Key petrophysical curves to analyze
    KEY_CURVES = ['PHIF', 'KLOGH', 'SW', 'VSH', 'BVW', 'PORD', 'KLOGV']
    
    def __init__(self):
        """Initialize the data processor."""
        pass
    
    def calculate_curve_statistics(self, log_data: pd.DataFrame, curve_name: str) -> Dict:
        """
        Calculate statistics for a specific curve.
        
        Args:
            log_data: DataFrame with log data
            curve_name: Name of the curve to analyze
            
        Returns:
            Dictionary with statistics
        """
        if curve_name not in log_data.columns:
            return {}
        
        curve_data = log_data[curve_name].dropna()
        
        if len(curve_data) == 0:
            return {}
        
        stats = {
            'mean': float(curve_data.mean()),
            'min': float(curve_data.min()),
            'max': float(curve_data.max()),
            'std': float(curve_data.std()),
            'median': float(curve_data.median()),
            'count': int(len(curve_data)),
            'p10': float(curve_data.quantile(0.10)),
            'p90': float(curve_data.quantile(0.90))
        }
        
        return stats
    
    def create_well_summary(self, well_data: Dict, formation_tops: Optional[List[Dict]] = None) -> Dict:
        """
        Create a comprehensive summary for a well.
        
        Args:
            well_data: Dictionary containing well data from LAS file
            formation_tops: List of formation tops for this well
            
        Returns:
            Dictionary with well summary
        """
        summary = {
            'well_name': well_data.get('well_name', 'Unknown'),
            'file_path': well_data.get('file_path', ''),
            'file_type': well_data.get('file_type', 'UNKNOWN'),
            'field': well_data.get('field', 'VOLVE'),
            'location': well_data.get('location', ''),
            'start_depth': well_data.get('start_depth'),
            'stop_depth': well_data.get('stop_depth'),
            'depth_range': None,
            'available_curves': well_data.get('curves', []),
            'curve_statistics': {},
            'formation_tops': formation_tops or [],
            'summary_text': ''
        }
        
        # Calculate depth range
        if summary['start_depth'] and summary['stop_depth']:
            summary['depth_range'] = summary['stop_depth'] - summary['start_depth']
        
        # Calculate statistics for key curves
        log_data = well_data.get('log_data', pd.DataFrame())
        if not log_data.empty:
            for curve in self.KEY_CURVES:
                # Try different case variations
                curve_variations = [curve, curve.upper(), curve.lower(), 
                                  curve.capitalize(), f'DEPTH.{curve}']
                
                found_curve = None
                for var in curve_variations:
                    if var in log_data.columns:
                        found_curve = var
                        break
                
                if found_curve:
                    stats = self.calculate_curve_statistics(log_data, found_curve)
                    if stats:
                        summary['curve_statistics'][curve] = stats
        
        # Generate summary text
        summary['summary_text'] = self._generate_summary_text(summary, log_data)
        
        return summary
    
    def _generate_summary_text(self, summary: Dict, log_data: pd.DataFrame) -> str:
        """
        Generate a text summary of the well for embedding.
        
        Args:
            summary: Well summary dictionary
            log_data: DataFrame with log data
            
        Returns:
            Text summary string
        """
        text_parts = []
        
        # Well identification
        text_parts.append(f"Well: {summary['well_name']}")
        if summary.get('field'):
            text_parts.append(f"Field: {summary['field']}")
        if summary.get('location'):
            text_parts.append(f"Location: {summary['location']}")
        
        # Depth information
        if summary.get('start_depth') and summary.get('stop_depth'):
            text_parts.append(
                f"Depth range: {summary['start_depth']:.1f}m to {summary['stop_depth']:.1f}m "
                f"({summary.get('depth_range', 0):.1f}m total)"
            )
        
        # Available curves
        if summary.get('available_curves'):
            text_parts.append(f"Available curves: {', '.join(summary['available_curves'][:10])}")
        
        # Key curve statistics
        if summary.get('curve_statistics'):
            stats_text = []
            for curve, stats in summary['curve_statistics'].items():
                if curve == 'PHIF' and stats:
                    stats_text.append(
                        f"Porosity (PHIF): mean={stats['mean']:.3f}, "
                        f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
                    )
                elif curve == 'KLOGH' and stats:
                    stats_text.append(
                        f"Permeability (KLOGH): mean={stats['mean']:.2f} mD, "
                        f"range=[{stats['min']:.2f}, {stats['max']:.2f}] mD"
                    )
                elif curve == 'SW' and stats:
                    stats_text.append(
                        f"Water saturation (SW): mean={stats['mean']:.3f}, "
                        f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
                    )
                elif curve == 'VSH' and stats:
                    stats_text.append(
                        f"Shale volume (VSH): mean={stats['mean']:.3f}, "
                        f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
                    )
            
            if stats_text:
                text_parts.append("Petrophysical properties: " + "; ".join(stats_text))
        
        # Formation information
        if summary.get('formation_tops'):
            formations = [f"{ft['formation_name']} at {ft['md']:.1f}m" 
                          for ft in summary['formation_tops'][:5]]
            if formations:
                text_parts.append(f"Formations: {', '.join(formations)}")
        
        return ". ".join(text_parts) + "."
    
    def create_depth_intervals(self, well_data: Dict, interval_size: float = 50.0) -> List[Dict]:
        """
        Create depth-interval based chunks for embedding.
        
        Args:
            well_data: Dictionary containing well data
            interval_size: Size of each depth interval in meters
            
        Returns:
            List of dictionaries with interval data
        """
        intervals = []
        log_data = well_data.get('log_data', pd.DataFrame())
        
        if log_data.empty:
            return intervals
        
        # Get depth column
        depth_col = None
        for col in log_data.columns:
            if 'DEPTH' in col.upper() or col.upper() == 'DEPTH':
                depth_col = col
                break
        
        if depth_col is None:
            return intervals
        
        start_depth = well_data.get('start_depth')
        stop_depth = well_data.get('stop_depth')
        
        if start_depth is None or stop_depth is None:
            # Try to infer from data
            if not log_data[depth_col].empty:
                start_depth = log_data[depth_col].min()
                stop_depth = log_data[depth_col].max()
            else:
                return intervals
        
        # Create intervals
        current_depth = start_depth
        well_name = well_data.get('well_name', 'Unknown')
        
        while current_depth < stop_depth:
            interval_end = min(current_depth + interval_size, stop_depth)
            
            # Filter data for this interval
            interval_data = log_data[
                (log_data[depth_col] >= current_depth) & 
                (log_data[depth_col] < interval_end)
            ]
            
            if not interval_data.empty:
                # Calculate statistics for this interval
                interval_stats = {}
                for curve in self.KEY_CURVES:
                    curve_variations = [curve, curve.upper(), curve.lower(), curve.capitalize()]
                    found_curve = None
                    for var in curve_variations:
                        if var in interval_data.columns:
                            found_curve = var
                            break
                    
                    if found_curve:
                        stats = self.calculate_curve_statistics(interval_data, found_curve)
                        if stats:
                            interval_stats[curve] = stats
                
                # Create interval description
                desc_parts = [
                    f"{well_name} interval {current_depth:.1f}m to {interval_end:.1f}m"
                ]
                
                if 'PHIF' in interval_stats:
                    phif = interval_stats['PHIF']
                    desc_parts.append(
                        f"porosity {phif['mean']:.3f} (range {phif['min']:.3f}-{phif['max']:.3f})"
                    )
                
                if 'KLOGH' in interval_stats:
                    klogh = interval_stats['KLOGH']
                    desc_parts.append(
                        f"permeability {klogh['mean']:.2f} mD"
                    )
                
                interval_dict = {
                    'well_name': well_name,
                    'start_depth': current_depth,
                    'end_depth': interval_end,
                    'statistics': interval_stats,
                    'description': ", ".join(desc_parts),
                    'data': interval_data
                }
                
                intervals.append(interval_dict)
            
            current_depth = interval_end
        
        return intervals
    
    def process_all_wells(self, all_well_data: List[Dict], 
                          formation_tops_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Process all wells and create summaries.
        
        Args:
            all_well_data: List of well data dictionaries from LAS files
            formation_tops_data: Dictionary mapping well names to formation tops
            
        Returns:
            List of well summary dictionaries
        """
        well_summaries = []
        
        for well_data in all_well_data:
            well_name = well_data.get('well_name', 'Unknown')
            
            # Get formation tops for this well
            formation_tops = None
            for key, formations in formation_tops_data.items():
                if well_name.upper() in key.upper() or key.upper() in well_name.upper():
                    formation_tops = formations
                    break
            
            # Create summary
            summary = self.create_well_summary(well_data, formation_tops)
            well_summaries.append(summary)
        
        return well_summaries
    
    def create_interval_chunks(self, all_well_data: List[Dict], 
                              interval_size: float = 50.0) -> List[Dict]:
        """
        Create depth interval chunks for all wells.
        
        Args:
            all_well_data: List of well data dictionaries
            interval_size: Size of each interval in meters
            
        Returns:
            List of interval dictionaries
        """
        all_intervals = []
        
        for well_data in all_well_data:
            intervals = self.create_depth_intervals(well_data, interval_size)
            all_intervals.extend(intervals)
        
        return all_intervals


if __name__ == "__main__":
    # Test the data processor
    from data_ingestion import LASFileReader
    from formation_tops_parser import FormationTopsParser
    
    # Load data
    reader = LASFileReader()
    all_wells = reader.process_all_wells()
    
    parser = FormationTopsParser()
    formation_tops = parser.parse_file()
    
    # Process data
    processor = WellDataProcessor()
    summaries = processor.process_all_wells(all_wells, formation_tops)
    
    if summaries:
        print(f"\nProcessed {len(summaries)} well summaries")
        print(f"\nSample summary text:")
        print(summaries[0]['summary_text'])

