"""
Parser for formation tops data from Well_picks_Volve_v1.dat file.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional


class FormationTopsParser:
    """Parses formation tops data from the Volve well picks file."""
    
    def __init__(self, file_path: str = "spwla_volve-main/Well_picks_Volve_v1.dat"):
        """
        Initialize the formation tops parser.
        
        Args:
            file_path: Path to the well picks data file
        """
        self.file_path = Path(file_path)
    
    def parse_file(self) -> Dict[str, List[Dict]]:
        """
        Parse the formation tops file.
        
        Returns:
            Dictionary mapping well names to lists of formation top records
        """
        if not self.file_path.exists():
            print(f"Warning: Formation tops file not found at {self.file_path}")
            return {}
        
        well_data = {}
        current_well = None
        
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Skip lines that are clearly header rows (even if they don't start with "Well")
            if 'well name' in line_lower and 'surface name' in line_lower:
                continue
            
            # Check if this is a well header line
            if line.startswith('Well '):
                # Extract well name
                match = re.match(r'Well\s+(.+)', line)
                if match:
                    well_name_candidate = match.group(1).strip()
                    # Skip if this looks like a header row (contains "name" and "Surface name")
                    if 'name' in well_name_candidate.lower() and 'surface name' in well_name_candidate.lower():
                        continue
                    current_well = well_name_candidate
                    well_data[current_well] = []
                continue
            
            # Check if this is a header row (contains column headers)
            # Header rows contain keywords like "Surface name", "Obs#", "MD", "TVD", "TVDSS" in sequence
            header_keywords = ['surface name', 'obs#', 'md', 'tvd', 'tvdss', 'twt', 'dip', 'azi', 'easting', 'northing', 'intrp']
            line_lower = line.lower()
            is_header = False
            
            # Check for exact header pattern: "Well name" followed by "Surface name"
            if 'well name' in line_lower and 'surface name' in line_lower:
                is_header = True
            # Check if this looks like a header row (contains multiple header keywords in sequence)
            elif 'surface name' in line_lower:
                keyword_count = sum(1 for kw in header_keywords if kw in line_lower)
                if keyword_count >= 3:  # Header rows typically have 3+ keywords
                    is_header = True
            # Check for lines that are just header keywords (no well name, no formation name, no numbers)
            elif any(kw in line_lower for kw in ['obs#', 'qlf']) and 'surface name' in line_lower:
                # This is likely a header row if it doesn't contain a well name or formation name
                if not any(part.isdigit() and 50 < float(part) < 5000 for part in line.split() if part.replace('.', '').isdigit()):
                    is_header = True
            
            # Check if this is a separator line (dashes)
            is_separator = line.startswith('-') and len(line) > 10 and line.count('-') > 5
            
            # Check if this is a data line (contains formation information)
            # Format appears to be: Well name (25 chars), Surface name (40 chars), Obs# (5), Qlf (3), MD (9), TVD (9), etc.
            if current_well and not is_separator and not is_header and not line.startswith('Well name'):
                try:
                    # The format is fixed-width based on the header
                    # Well name: ~25 chars, Surface name: ~40 chars, then numeric fields
                    # Use regex to extract the formation name and depths
                    
                    # Pattern: well name (already known), then formation name, then numbers
                    # Formation name is typically between the well name and the first significant number
                    parts = line.split()
                    
                    if len(parts) < 3:
                        continue
                    
                    # Find where the numeric values start (MD, TVD, etc.)
                    # These are typically after the formation name
                    md_value = None
                    tvd_value = None
                    tvdss_value = None
                    formation_name = None
                    
                    # Look for the pattern: well name (may repeat), formation name, then numbers
                    # The formation name is usually the longest text field before numbers
                    numeric_start_idx = None
                    for i, part in enumerate(parts):
                        try:
                            # Check if this is a numeric value (could be MD)
                            val = float(part)
                            # MD values are typically in reasonable depth range
                            if 50 < val < 5000:
                                numeric_start_idx = i
                                md_value = val
                                break
                        except ValueError:
                            continue
                    
                    if numeric_start_idx is not None and numeric_start_idx > 1:
                        # Formation name should be between well name and first number
                        # Skip the well name (first part, might be repeated)
                        formation_parts = []
                        start_idx = 1
                        
                        # Skip well name if it appears again
                        if len(parts) > 1 and parts[1].upper() == current_well.split()[-1].upper():
                            start_idx = 2
                        
                        # Collect formation name parts until we hit numbers
                        for i in range(start_idx, numeric_start_idx):
                            formation_parts.append(parts[i])
                        
                        if formation_parts:
                            formation_name = ' '.join(formation_parts).strip()
                        
                        # Try to get TVD and TVDSS (next numeric values after MD)
                        if numeric_start_idx + 1 < len(parts):
                            try:
                                tvd_value = float(parts[numeric_start_idx + 1])
                            except (ValueError, IndexError):
                                pass
                        
                        if numeric_start_idx + 2 < len(parts):
                            try:
                                tvdss_value = float(parts[numeric_start_idx + 2])
                            except (ValueError, IndexError):
                                pass
                    
                    # Alternative: if we can't find MD, try to extract formation name differently
                    if formation_name is None:
                        # Look for common formation keywords
                        for i in range(1, len(parts) - 2):
                            potential = ' '.join(parts[1:i+1])
                            if any(kw in potential for kw in ['Fm.', 'Top', 'Base', 'GP.', 'Formation', 'Seabed']):
                                formation_name = potential
                                # Try to find MD after this
                                if i + 1 < len(parts):
                                    try:
                                        md_value = float(parts[i + 1])
                                    except (ValueError, IndexError):
                                        pass
                                break
                    
                    # Validate formation name - skip if it's clearly a header row
                    if formation_name and md_value is not None:
                        formation_name_clean = formation_name.strip()
                        # Skip if formation name contains too many header keywords (likely a header row)
                        header_keyword_count = sum(1 for kw in header_keywords if kw in formation_name_clean.lower())
                        # Skip if formation name is too long and contains header keywords (like "name Surface name Obs# Qlf MD TVD TVDSS TWT Dip Azi Easting Northing Intrp")
                        if header_keyword_count >= 3 and len(formation_name_clean) > 50:
                            continue
                        # Skip if formation name is exactly a header pattern
                        if formation_name_clean.lower().startswith('name surface name'):
                            continue
                        
                        formation_record = {
                            'well_name': current_well,
                            'formation_name': formation_name_clean,
                            'md': md_value,
                            'tvd': tvd_value,
                            'tvdss': tvdss_value
                        }
                        well_data[current_well].append(formation_record)
                
                except Exception as e:
                    # Skip lines that can't be parsed
                    continue
        
        return well_data
    
    def get_formations_for_well(self, well_name: str, well_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Get formation tops for a specific well.
        
        Args:
            well_name: Name of the well
            well_data: Parsed formation tops data
            
        Returns:
            List of formation records for the well
        """
        # Try exact match first
        if well_name in well_data:
            return well_data[well_name]
        
        # Try partial match (well name might be formatted differently)
        for key, formations in well_data.items():
            if well_name.upper() in key.upper() or key.upper() in well_name.upper():
                return formations
        
        return []
    
    def get_all_formations(self, well_data: Dict[str, List[Dict]]) -> List[str]:
        """
        Get list of all unique formation names.
        
        Args:
            well_data: Parsed formation tops data
            
        Returns:
            List of unique formation names
        """
        formations = set()
        for well_formations in well_data.values():
            for formation in well_formations:
                formations.add(formation['formation_name'])
        return sorted(list(formations))


if __name__ == "__main__":
    # Test the formation tops parser
    parser = FormationTopsParser()
    well_data = parser.parse_file()
    
    print(f"Parsed formation tops for {len(well_data)} wells")
    
    if well_data:
        # Show first well's formations
        first_well = list(well_data.keys())[0]
        print(f"\nFormations for {first_well}:")
        for formation in well_data[first_well][:5]:
            print(f"  {formation['formation_name']} at MD: {formation['md']}")

