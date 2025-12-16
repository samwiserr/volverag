"""
LAS file I/O module following Geolog-Python-Loglan patterns.
Provides clean interface for reading LAS files and extracting data.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import lasio
import pandas as pd
import numpy as np


class CurveMetadata:
    """Metadata for a log curve."""
    
    def __init__(self, mnemonic: str, unit: str = "", description: str = "", 
                 data: Optional[pd.Series] = None):
        self.mnemonic = mnemonic
        self.unit = unit
        self.description = description
        self.data = data
        self.stats = self._compute_stats() if data is not None else {}
    
    def _compute_stats(self) -> Dict:
        """Compute basic statistics for the curve."""
        if self.data is None or len(self.data) == 0:
            return {}
        
        clean_data = self.data.dropna()
        if len(clean_data) == 0:
            return {}
        
        return {
            'min': float(clean_data.min()),
            'max': float(clean_data.max()),
            'mean': float(clean_data.mean()),
            'std': float(clean_data.std()),
            'count': int(len(clean_data)),
            'null_count': int(len(self.data) - len(clean_data))
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mnemonic': self.mnemonic,
            'unit': self.unit,
            'description': self.description,
            'stats': self.stats
        }


def load_las(path: Union[str, Path], return_dataframe: bool = True) -> Union[lasio.LASFile, pd.DataFrame, tuple]:
    """
    Load a LAS file from disk.
    
    Args:
        path: Path to LAS file
        return_dataframe: If True, also return DataFrame. If False, return only LASFile.
                         If 'both', return tuple (LASFile, DataFrame)
    
    Returns:
        LASFile object, DataFrame, or tuple depending on return_dataframe parameter
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"LAS file not found: {path}")
    
    try:
        # Read LAS file - handle wrapped files
        las = lasio.read(str(path), engine='normal')
        
        if return_dataframe is False:
            return las
        elif return_dataframe == 'both':
            df = las.df()
            # Handle null values
            null_value = -999.25
            if hasattr(las, 'well') and las.well:
                for item in las.well:
                    if item.mnemonic.upper() == 'NULL':
                        try:
                            null_value = float(item.value)
                        except:
                            pass
                        break
            df = df.replace(null_value, pd.NA)
            return las, df
        else:
            # Return DataFrame only
            df = las.df()
            # Handle null values
            null_value = -999.25
            if hasattr(las, 'well') and las.well:
                for item in las.well:
                    if item.mnemonic.upper() == 'NULL':
                        try:
                            null_value = float(item.value)
                        except:
                            pass
                        break
            df = df.replace(null_value, pd.NA)
            return df
    
    except Exception as e:
        raise RuntimeError(f"Error reading LAS file {path}: {e}") from e


def list_curves(las: lasio.LASFile, include_stats: bool = False) -> List[CurveMetadata]:
    """
    List all curves in a LAS file with metadata.
    
    Args:
        las: LASFile object
        include_stats: If True, compute statistics for each curve
    
    Returns:
        List of CurveMetadata objects
    """
    curves = []
    
    if not hasattr(las, 'curves') or not las.curves:
        return curves
    
    # Get DataFrame for stats if needed
    df = None
    if include_stats:
        try:
            df = las.df()
        except:
            pass
    
    for curve in las.curves:
        mnemonic = curve.mnemonic
        unit = curve.unit if hasattr(curve, 'unit') else ""
        description = curve.descr if hasattr(curve, 'descr') else ""
        
        # Get data for stats if requested
        data = None
        if include_stats and df is not None and mnemonic in df.columns:
            data = df[mnemonic]
        
        curve_meta = CurveMetadata(
            mnemonic=mnemonic,
            unit=unit,
            description=description,
            data=data
        )
        curves.append(curve_meta)
    
    return curves


def get_interval_df(las: lasio.LASFile, depth_top: float, depth_base: float, 
                   curves: Optional[List[str]] = None, depth_col: Optional[str] = None) -> pd.DataFrame:
    """
    Extract a depth interval from LAS file as DataFrame.
    
    Args:
        las: LASFile object
        depth_top: Top depth (MD) in meters
        depth_base: Base depth (MD) in meters
        curves: Optional list of curve names to include. If None, includes all curves.
        depth_col: Name of depth column. If None, auto-detects (DEPTH, MD, etc.)
    
    Returns:
        DataFrame with data for the specified interval
    """
    # Get DataFrame
    df = las.df()
    
    # Find depth column
    if depth_col is None:
        depth_col = None
        for col in df.columns:
            col_upper = col.upper()
            if 'DEPTH' in col_upper or col_upper == 'MD' or col_upper == 'TVD':
                depth_col = col
                break
        
        if depth_col is None:
            # Try first column if it looks numeric
            first_col = df.columns[0]
            if df[first_col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                depth_col = first_col
    
    if depth_col is None or depth_col not in df.columns:
        raise ValueError(f"Could not find depth column in LAS file. Available columns: {list(df.columns)}")
    
    # Filter by depth
    mask = (df[depth_col] >= depth_top) & (df[depth_col] <= depth_base)
    interval_df = df[mask].copy()
    
    # Filter by curves if specified
    if curves:
        # Normalize curve names (handle case variations)
        available_curves = []
        for curve in curves:
            # Try exact match first
            if curve in interval_df.columns:
                available_curves.append(curve)
            else:
                # Try case-insensitive match
                for col in interval_df.columns:
                    if col.upper() == curve.upper():
                        available_curves.append(col)
                        break
        
        # Always include depth column
        if depth_col not in available_curves:
            available_curves.insert(0, depth_col)
        
        interval_df = interval_df[available_curves]
    
    return interval_df


def get_well_metadata(las: lasio.LASFile) -> Dict:
    """
    Extract well metadata from LAS file header.
    
    Args:
        las: LASFile object
    
    Returns:
        Dictionary with well metadata
    """
    metadata = {}
    
    if hasattr(las, 'well') and las.well:
        for item in las.well:
            mnemonic = item.mnemonic.upper()
            value = item.value
            
            if mnemonic == 'WELL':
                metadata['well_name'] = str(value) if value else None
            elif mnemonic == 'FLD':
                metadata['field'] = str(value) if value else None
            elif mnemonic == 'LOC':
                metadata['location'] = str(value) if value else None
            elif mnemonic == 'UWI':
                metadata['uwi'] = str(value) if value else None
            elif mnemonic == 'COMP':
                metadata['company'] = str(value) if value else None
            elif mnemonic == 'SRVC':
                metadata['service_company'] = str(value) if value else None
            elif mnemonic == 'STRT':
                metadata['start_depth'] = float(value) if value else None
            elif mnemonic == 'STOP':
                metadata['stop_depth'] = float(value) if value else None
            elif mnemonic == 'STEP':
                metadata['step'] = float(value) if value else None
            elif mnemonic == 'NULL':
                metadata['null_value'] = float(value) if value else -999.25
            elif mnemonic == 'KB':
                metadata['kb'] = float(value) if value else None
            elif mnemonic == 'DF':
                metadata['df'] = float(value) if value else None
            elif mnemonic == 'DATE':
                metadata['date'] = str(value) if value else None
    
    # Get curve list
    if hasattr(las, 'curves') and las.curves:
        metadata['curves'] = [c.mnemonic for c in las.curves]
    
    return metadata


if __name__ == "__main__":
    # Test the las_io module
    test_path = Path("spwla_volve-main/15_9-F-1/WLC_PETRO_COMPUTED_OUTPUT_1.LAS")
    
    if test_path.exists():
        print("Testing LAS I/O module...")
        
        # Test load_las
        las, df = load_las(test_path, return_dataframe='both')
        print(f"Loaded LAS file: {len(df)} rows, {len(df.columns)} columns")
        
        # Test list_curves
        curves = list_curves(las, include_stats=True)
        print(f"Found {len(curves)} curves")
        for curve in curves[:5]:
            print(f"  - {curve.mnemonic}: {curve.unit} ({curve.description})")
        
        # Test get_interval_df
        if 'DEPTH' in df.columns:
            depth_min = df['DEPTH'].min()
            depth_max = df['DEPTH'].max()
            interval = get_interval_df(las, depth_min, depth_min + 100, curves=['GR', 'PHIF'])
            print(f"Interval DataFrame: {len(interval)} rows")
        
        # Test get_well_metadata
        metadata = get_well_metadata(las)
        print(f"Well metadata: {metadata.get('well_name', 'Unknown')}")
    else:
        print(f"Test file not found: {test_path}")

