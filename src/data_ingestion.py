"""
Data ingestion module for reading LAS files from Volve dataset.
Uses lasio library to parse LAS files and extract well metadata and curve information.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import lasio
import pandas as pd
import numpy as np


class LASFileReader:
    """Reads and parses LAS files from the Volve dataset."""
    
    def __init__(self, base_path: str = "spwla_volve-main"):
        """
        Initialize the LAS file reader.
        
        Args:
            base_path: Base path to the Volve dataset directory
        """
        self.base_path = Path(base_path)
        self.well_folders = self._discover_well_folders()
    
    def _discover_well_folders(self) -> List[Path]:
        """Discover all well folders in the dataset."""
        well_folders = []
        if not self.base_path.exists():
            return well_folders
        
        # Look for folders that might contain well data
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if folder contains LAS files
                las_files = list(item.glob("*.las")) + list(item.glob("*.LAS"))
                if las_files:
                    well_folders.append(item)
        
        return well_folders
    
    def read_las_file(self, file_path: Path) -> Optional[lasio.LASFile]:
        """
        Read a single LAS file.
        
        Args:
            file_path: Path to the LAS file
            
        Returns:
            LASFile object or None if reading fails
        """
        try:
            las = lasio.read(str(file_path))
            return las
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def extract_well_metadata(self, las: lasio.LASFile, file_path: Path) -> Dict:
        """
        Extract well metadata from LAS file header.
        
        Args:
            las: LASFile object
            file_path: Path to the LAS file
            
        Returns:
            Dictionary containing well metadata
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'well_folder': file_path.parent.name,
        }
        
        # Extract well information from header
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
                elif mnemonic == 'STRT':
                    metadata['start_depth'] = float(value) if value else None
                elif mnemonic == 'STOP':
                    metadata['stop_depth'] = float(value) if value else None
                elif mnemonic == 'STEP':
                    metadata['step'] = float(value) if value else None
                elif mnemonic == 'NULL':
                    metadata['null_value'] = float(value) if value else -999.25
                elif mnemonic == 'COMP':
                    metadata['company'] = str(value) if value else None
                elif mnemonic == 'LATI':
                    metadata['latitude'] = str(value) if value else None
                elif mnemonic == 'LONG':
                    metadata['longitude'] = str(value) if value else None
        
        # If well name not found, try to infer from folder name
        if 'well_name' not in metadata or not metadata['well_name']:
            metadata['well_name'] = file_path.parent.name
        
        return metadata
    
    def extract_curve_info(self, las: lasio.LASFile) -> Dict:
        """
        Extract curve information from LAS file.
        
        Args:
            las: LASFile object
            
        Returns:
            Dictionary containing curve information
        """
        curve_info = {
            'curves': [],
            'curve_descriptions': {},
            'curve_units': {}
        }
        
        if hasattr(las, 'curves') and las.curves:
            for curve in las.curves:
                mnemonic = curve.mnemonic
                unit = curve.unit if hasattr(curve, 'unit') else None
                description = curve.descr if hasattr(curve, 'descr') else None
                
                curve_info['curves'].append(mnemonic)
                if description:
                    curve_info['curve_descriptions'][mnemonic] = description
                if unit:
                    curve_info['curve_units'][mnemonic] = unit
        
        return curve_info
    
    def extract_log_data(self, las: lasio.LASFile, null_value: float = -999.25) -> pd.DataFrame:
        """
        Extract log data as pandas DataFrame.
        
        Args:
            las: LASFile object
            null_value: Value to use for null/missing data
            
        Returns:
            DataFrame with log data
        """
        try:
            df = las.df()
            # Replace null values
            if null_value is not None:
                df = df.replace(null_value, np.nan)
            return df
        except Exception as e:
            print(f"Error extracting log data: {e}")
            return pd.DataFrame()
    
    def process_well_folder(self, well_folder: Path) -> List[Dict]:
        """
        Process all LAS files in a well folder.
        
        Args:
            well_folder: Path to well folder
            
        Returns:
            List of dictionaries containing processed well data
        """
        well_data_list = []
        
        # Find all LAS files in the folder
        las_files = list(well_folder.glob("*.las")) + list(well_folder.glob("*.LAS"))
        
        for las_file in las_files:
            las = self.read_las_file(las_file)
            if las is None:
                continue
            
            # Extract metadata
            metadata = self.extract_well_metadata(las, las_file)
            
            # Extract curve information
            curve_info = self.extract_curve_info(las)
            
            # Extract log data
            log_data = self.extract_log_data(las, metadata.get('null_value', -999.25))
            
            # Combine all information
            well_data = {
                **metadata,
                **curve_info,
                'log_data': log_data,
                'file_type': self._classify_file_type(las_file.name)
            }
            
            well_data_list.append(well_data)
        
        return well_data_list
    
    def _classify_file_type(self, filename: str) -> str:
        """Classify the type of LAS file based on filename."""
        filename_upper = filename.upper()
        if 'CPI' in filename_upper:
            return 'CPI'
        elif 'LFP' in filename_upper:
            return 'LFP'
        elif 'INPUT' in filename_upper:
            return 'INPUT'
        elif 'OUTPUT' in filename_upper:
            return 'OUTPUT'
        elif 'COMPOSITE' in filename_upper:
            return 'COMPOSITE'
        else:
            return 'UNKNOWN'
    
    def process_all_wells(self) -> List[Dict]:
        """
        Process all wells in the dataset.
        
        Returns:
            List of dictionaries containing processed well data for all wells
        """
        all_well_data = []
        
        print(f"Processing {len(self.well_folders)} well folders...")
        for well_folder in self.well_folders:
            print(f"Processing {well_folder.name}...")
            well_data_list = self.process_well_folder(well_folder)
            all_well_data.extend(well_data_list)
        
        print(f"Processed {len(all_well_data)} LAS files from {len(self.well_folders)} well folders")
        return all_well_data
    
    def get_available_curves(self, all_well_data: List[Dict]) -> Dict[str, List[str]]:
        """
        Get list of available curves across all wells.
        
        Args:
            all_well_data: List of processed well data dictionaries
            
        Returns:
            Dictionary mapping well names to available curves
        """
        curve_map = {}
        for well_data in all_well_data:
            well_name = well_data.get('well_name', 'Unknown')
            curves = well_data.get('curves', [])
            if well_name not in curve_map:
                curve_map[well_name] = []
            curve_map[well_name].extend(curves)
        
        # Remove duplicates
        for well_name in curve_map:
            curve_map[well_name] = list(set(curve_map[well_name]))
        
        return curve_map


if __name__ == "__main__":
    # Test the LAS file reader
    reader = LASFileReader()
    all_wells = reader.process_all_wells()
    
    if all_wells:
        print(f"\nSuccessfully processed {len(all_wells)} LAS files")
        print(f"\nSample well metadata:")
        print(all_wells[0].keys())
        
        # Show available curves for first well
        if 'curves' in all_wells[0]:
            print(f"\nAvailable curves in first well: {all_wells[0]['curves'][:10]}")

