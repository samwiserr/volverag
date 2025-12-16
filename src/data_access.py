"""
Data access layer for on-demand loading of LAS data and formation information.
Provides quick access to well log data for visualization and computation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.data_ingestion import LASFileReader
from src.formation_tops_parser import FormationTopsParser
from src.las_io import load_las


class DataAccess:
    """Provides on-demand access to LAS data and formation information."""
    
    def __init__(self, base_path: str = "spwla_volve-main"):
        """
        Initialize the data access layer.
        
        Args:
            base_path: Base path to the Volve dataset directory
        """
        self.base_path = Path(base_path)
        self.reader = LASFileReader(base_path)
        self.formation_parser = FormationTopsParser(f"{base_path}/Well_picks_Volve_v1.dat")
        
        # Cache for loaded data
        self._log_data_cache: Dict[str, pd.DataFrame] = {}
        self._formation_tops_cache: Optional[Dict[str, List[Dict]]] = None
    
    def get_formation_tops(self) -> Dict[str, List[Dict]]:
        """Get formation tops data (cached)."""
        if self._formation_tops_cache is None:
            self._formation_tops_cache = self.formation_parser.parse_file()
        return self._formation_tops_cache
    
    def find_well_las_files(self, well_name: str) -> List[Path]:
        """
        Find LAS files for a specific well.
        
        Args:
            well_name: Name of the well
            
        Returns:
            List of paths to LAS files for the well
        """
        las_files = []
        well_folders = self.reader._discover_well_folders()
        
        # Normalize well name for matching
        well_name_normalized = well_name.upper().replace(' ', '_').replace('/', '-')
        
        for well_folder in well_folders:
            folder_name_normalized = well_folder.name.upper().replace(' ', '_').replace('/', '-')
            
            # Check if well name matches folder
            if (well_name_normalized in folder_name_normalized or 
                folder_name_normalized in well_name_normalized):
                # Find LAS files in this folder
                folder_las = list(well_folder.glob("*.las")) + list(well_folder.glob("*.LAS"))
                las_files.extend(folder_las)
        
        return las_files
    
    def load_well_log_data(self, well_name: str, prefer_output: bool = True) -> Optional[pd.DataFrame]:
        """
        Load log data for a specific well.
        
        Args:
            well_name: Name of the well
            prefer_output: If True, prefer computed output files
            
        Returns:
            DataFrame with log data or None if not found
        """
        cache_key = f"{well_name}_{prefer_output}"
        if cache_key in self._log_data_cache:
            return self._log_data_cache[cache_key]
        
        las_files = self.find_well_las_files(well_name)
        if not las_files:
            return None
        
        # Prefer output files if requested
        if prefer_output:
            output_files = [f for f in las_files if 'OUTPUT' in f.name.upper() or 'CPI' in f.name.upper()]
            if output_files:
                las_files = output_files
        
        # Try to load the first available LAS file using las_io module
        for las_file in las_files:
            try:
                # Use las_io.load_las to get DataFrame directly
                df = load_las(las_file, return_dataframe=True)
                
                if df is not None and not df.empty:
                    # Cache the result
                    self._log_data_cache[cache_key] = df
                    return df
            except Exception as e:
                print(f"Error loading {las_file}: {e}")
                continue
        
        return None
    
    def get_formation_depth_range(self, well_name: str, formation_name: str) -> Optional[Tuple[float, float]]:
        """
        Get the depth range (top and base) for a formation in a well.
        
        Args:
            well_name: Name of the well
            formation_name: Name of the formation
            
        Returns:
            Tuple of (top_depth, base_depth) in MD, or None if not found
        """
        formation_tops = self.get_formation_tops()
        
        # Find well in formation tops
        well_formations = None
        for key, formations in formation_tops.items():
            if well_name.upper() in key.upper() or key.upper() in well_name.upper():
                well_formations = formations
                break
        
        if not well_formations:
            return None
        
        # Find formation top
        formation_top = None
        formation_base = None
        
        for ft in well_formations:
            ft_name = ft.get('formation_name', '').lower()
            formation_lower = formation_name.lower()
            
            # Check for exact match or partial match
            if formation_lower in ft_name or ft_name in formation_lower:
                # Check if this is a base
                if 'base' in ft_name:
                    formation_base = ft.get('md')
                else:
                    formation_top = ft.get('md')
        
        # If we found top but no base, look for next formation or "Base" entry
        if formation_top and not formation_base:
            # Look for "Base" entry for this formation
            for ft in well_formations:
                ft_name = ft.get('formation_name', '').lower()
                if 'base' in ft_name and formation_name.lower() in ft_name:
                    formation_base = ft.get('md')
                    break
            
            # If still no base, use next formation's top as base
            if not formation_base:
                found_top = False
                for ft in well_formations:
                    if found_top:
                        formation_base = ft.get('md')
                        break
                    ft_name = ft.get('formation_name', '').lower()
                    if formation_name.lower() in ft_name:
                        found_top = True
        
        if formation_top:
            return (formation_top, formation_base) if formation_base else (formation_top, None)
        
        return None
    
    def filter_log_by_depth(self, log_data: pd.DataFrame, start_depth: float, 
                           end_depth: Optional[float] = None, depth_col: str = "DEPTH") -> pd.DataFrame:
        """
        Filter log data by depth range.
        
        Args:
            log_data: DataFrame with log data
            start_depth: Start depth (MD)
            end_depth: End depth (MD), if None uses max depth
            depth_col: Name of depth column
            
        Returns:
            Filtered DataFrame
        """
        if log_data.empty or depth_col not in log_data.columns:
            return pd.DataFrame()
        
        if end_depth is None:
            filtered = log_data[log_data[depth_col] >= start_depth]
        else:
            filtered = log_data[
                (log_data[depth_col] >= start_depth) & 
                (log_data[depth_col] <= end_depth)
            ]
        
        return filtered
    
    def get_available_curves(self, well_name: str) -> List[str]:
        """
        Get list of available curves for a well.
        
        Args:
            well_name: Name of the well
            
        Returns:
            List of curve names
        """
        log_data = self.load_well_log_data(well_name)
        if log_data is None or log_data.empty:
            return []
        
        return list(log_data.columns)
    
    def get_curve_data(self, well_name: str, curve_name: str, 
                      depth_range: Optional[Tuple[float, float]] = None) -> Optional[pd.Series]:
        """
        Get specific curve data for a well, optionally filtered by depth.
        
        Args:
            well_name: Name of the well
            curve_name: Name of the curve
            depth_range: Optional (start_depth, end_depth) tuple
            
        Returns:
            Series with curve data or None
        """
        log_data = self.load_well_log_data(well_name)
        if log_data is None or log_data.empty:
            return None
        
        # Try different case variations
        curve_variations = [curve_name, curve_name.upper(), curve_name.lower(), 
                          curve_name.capitalize(), f'DEPTH.{curve_name}']
        
        found_curve = None
        for var in curve_variations:
            if var in log_data.columns:
                found_curve = var
                break
        
        if not found_curve:
            return None
        
        curve_data = log_data[found_curve].copy()
        
        # Filter by depth if specified
        if depth_range:
            depth_col = None
            for col in log_data.columns:
                if 'DEPTH' in col.upper():
                    depth_col = col
                    break
            
            if depth_col:
                start_depth, end_depth = depth_range
                mask = log_data[depth_col] >= start_depth
                if end_depth:
                    mask = mask & (log_data[depth_col] <= end_depth)
                curve_data = curve_data[mask]
        
        return curve_data


if __name__ == "__main__":
    # Test the data access layer
    access = DataAccess()
    
    # Test loading well data
    well_name = "15/9-F-1"
    log_data = access.load_well_log_data(well_name)
    print(f"Loaded data for {well_name}: {log_data.shape if log_data is not None else 'Not found'}")
    
    # Test formation depth range
    depth_range = access.get_formation_depth_range(well_name, "Hugin")
    print(f"Hugin formation depth range: {depth_range}")
    
    # Test available curves
    curves = access.get_available_curves(well_name)
    print(f"Available curves: {curves[:10]}")

