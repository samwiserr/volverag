"""
Computational tools for on-the-fly analysis and visualization.
These tools can be called by the LLM to perform computations and generate visualizations.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_access import DataAccess
from src.las_io import load_las, get_interval_df, list_curves
from src.plots import depth_plot, crossplot, formation_depth_plot


class ComputationalTools:
    """Collection of computational tools for well data analysis."""
    
    def __init__(self, data_access: DataAccess, formation_tops_data: Optional[Dict] = None):
        """
        Initialize computational tools.
        
        Args:
            data_access: DataAccess instance for loading data
            formation_tops_data: Optional dictionary mapping well names to formation tops
        """
        self.data_access = data_access
        self.formation_tops_data = formation_tops_data or {}
    
    def get_tool_schemas(self) -> List[Dict]:
        """
        Get schemas for all available tools (for LLM function calling).
        
        Returns:
            List of tool schema dictionaries
        """
        return [
            {
                "name": "plot_formation_log",
                "description": "Plot well log curves with a specific formation interval highlighted. Use this when user asks about formations or wants to visualize formation data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "well_name": {
                            "type": "string",
                            "description": "Name of the well (e.g., '15/9-F-1' or 'NO_15/9-19_A')"
                        },
                        "formation_name": {
                            "type": "string",
                            "description": "Name of the formation (e.g., 'Hugin', 'Sleipner', 'Hugin Fm. VOLVE Top')"
                        },
                        "curves": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of curves to plot (e.g., ['GR', 'PHIF', 'VSH']). For shale volume questions, include 'GR'. For porosity, include 'PHIF'."
                        }
                    },
                    "required": ["well_name", "formation_name", "curves"]
                }
            },
            {
                "name": "get_formation_interval",
                "description": "Get the depth range (top and base) for a formation in a well.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "well_name": {"type": "string", "description": "Name of the well"},
                        "formation_name": {"type": "string", "description": "Name of the formation"}
                    },
                    "required": ["well_name", "formation_name"]
                }
            },
            {
                "name": "calculate_formation_statistics",
                "description": "Calculate statistics (mean, min, max, std) for a curve within a formation interval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "well_name": {"type": "string", "description": "Name of the well"},
                        "formation_name": {"type": "string", "description": "Name of the formation"},
                        "curve": {"type": "string", "description": "Name of the curve (e.g., 'PHIF', 'KLOGH', 'VSH', 'GR')"}
                    },
                    "required": ["well_name", "formation_name", "curve"]
                }
            },
            {
                "name": "plot_log_curves",
                "description": "Plot multiple log curves for a well over a specified depth range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "well_name": {"type": "string", "description": "Name of the well"},
                        "curves": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of curves to plot"
                        },
                        "depth_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "number", "description": "Start depth in meters"},
                                "end": {"type": "number", "description": "End depth in meters"}
                            },
                            "description": "Optional depth range"
                        }
                    },
                    "required": ["well_name", "curves"]
                }
            },
            {
                "name": "get_relevant_curves",
                "description": "Get suggested relevant curves based on a query about a specific property.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "property": {
                            "type": "string",
                            "description": "The property being queried (e.g., 'shale volume', 'porosity', 'permeability', 'water saturation')"
                        }
                    },
                    "required": ["property"]
                }
            },
            {
                "name": "list_all_formations",
                "description": "Get a complete list of all unique formation names from all wells in the dataset. Use this for queries asking to 'list all formations' or 'list all available surfaces'.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "list_formations_by_well",
                "description": "Get all formations for a specific well. Use this when the user asks about formations in a particular well.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "well_name": {
                            "type": "string",
                            "description": "Name of the well (e.g., 'NO 15/9-11', '15/9-F-1', 'NO_15/9-19_A')"
                        }
                    },
                    "required": ["well_name"]
                }
            },
            {
                "name": "list_all_wells_with_formations",
                "description": "Get a complete mapping of all wells to their formations. Use this for queries asking to 'list all formations in all wells' or similar comprehensive listing requests.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    
    def plot_formation_log(self, well_name: str, formation_name: str, 
                          curves: List[str]) -> Dict[str, Any]:
        """
        Plot well log curves with formation interval highlighted.
        
        Args:
            well_name: Name of the well
            formation_name: Name of the formation
            curves: List of curves to plot
            
        Returns:
            Dictionary with chart data and metadata
        """
        # Load log data
        log_data = self.data_access.load_well_log_data(well_name)
        if log_data is None or log_data.empty:
            return {
                "success": False,
                "error": f"Could not load log data for well {well_name}"
            }
        
        # Get formation depth range
        depth_range = self.data_access.get_formation_depth_range(well_name, formation_name)
        if not depth_range:
            return {
                "success": False,
                "error": f"Could not find formation {formation_name} in well {well_name}"
            }
        
        formation_top, formation_base = depth_range
        
        # Find depth column
        depth_col = None
        for col in log_data.columns:
            if 'DEPTH' in col.upper() or col.upper() == 'DEPTH':
                depth_col = col
                break
        
        if not depth_col:
            return {
                "success": False,
                "error": "Could not find depth column in log data"
            }
        
        # Filter data to reasonable range (formation ± 100m for context)
        context_margin = 100.0
        plot_start = max(formation_top - context_margin, log_data[depth_col].min())
        plot_end = min((formation_base if formation_base else log_data[depth_col].max()) + context_margin,
                      log_data[depth_col].max())
        
        plot_data = self.data_access.filter_log_by_depth(log_data, plot_start, plot_end, depth_col)
        
        # Find available curves
        available_curves = []
        for curve in curves:
            curve_variations = [curve, curve.upper(), curve.lower(), curve.capitalize()]
            for var in curve_variations:
                if var in plot_data.columns:
                    available_curves.append(var)
                    break
        
        if not available_curves:
            return {
                "success": False,
                "error": f"None of the requested curves {curves} are available in the log data"
            }
        
        # Create visualization with formation highlighting using plots module
        chart = formation_depth_plot(
            df=plot_data,
            curves=available_curves,
            formation_top=formation_top,
            formation_base=formation_base,
            depth_col=depth_col,
            well_name=well_name,
            formation_name=formation_name
        )
        
        return {
            "success": True,
            "chart": chart,
            "well_name": well_name,
            "formation_name": formation_name,
            "formation_top": formation_top,
            "formation_base": formation_base,
            "curves": available_curves,
            "depth_range": (plot_start, plot_end)
        }
    
    def get_formation_interval(self, well_name: str, formation_name: str) -> Dict[str, Any]:
        """
        Get the depth range for a formation in a well.
        
        Args:
            well_name: Name of the well
            formation_name: Name of the formation
            
        Returns:
            Dictionary with depth range information
        """
        depth_range = self.data_access.get_formation_depth_range(well_name, formation_name)
        
        if not depth_range:
            return {
                "success": False,
                "error": f"Could not find formation {formation_name} in well {well_name}"
            }
        
        formation_top, formation_base = depth_range
        
        return {
            "success": True,
            "well_name": well_name,
            "formation_name": formation_name,
            "top_depth": formation_top,
            "base_depth": formation_base,
            "thickness": (formation_base - formation_top) if formation_base else None
        }
    
    def calculate_formation_statistics(self, well_name: str, formation_name: str, 
                                     curve: str) -> Dict[str, Any]:
        """
        Calculate statistics for a curve within a formation interval.
        
        Args:
            well_name: Name of the well
            formation_name: Name of the formation
            curve: Name of the curve
            
        Returns:
            Dictionary with statistics
        """
        # Get formation depth range
        depth_range = self.data_access.get_formation_depth_range(well_name, formation_name)
        if not depth_range:
            return {
                "success": False,
                "error": f"Could not find formation {formation_name} in well {well_name}"
            }
        
        # Get curve data for formation interval
        curve_data = self.data_access.get_curve_data(well_name, curve, depth_range)
        
        if curve_data is None or curve_data.empty:
            return {
                "success": False,
                "error": f"Could not load curve {curve} for well {well_name}"
            }
        
        # Remove null values
        curve_data_clean = curve_data.dropna()
        
        if len(curve_data_clean) == 0:
            return {
                "success": False,
                "error": f"No valid data points for curve {curve} in formation {formation_name}"
            }
        
        # Calculate statistics
        stats = {
            "mean": float(curve_data_clean.mean()),
            "min": float(curve_data_clean.min()),
            "max": float(curve_data_clean.max()),
            "std": float(curve_data_clean.std()),
            "median": float(curve_data_clean.median()),
            "count": int(len(curve_data_clean))
        }
        
        return {
            "success": True,
            "well_name": well_name,
            "formation_name": formation_name,
            "curve": curve,
            "depth_range": depth_range,
            "statistics": stats
        }
    
    def plot_log_curves(self, well_name: str, curves: List[str], 
                       depth_range: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Plot log curves for a well.
        
        Args:
            well_name: Name of the well
            curves: List of curves to plot
            depth_range: Optional dict with 'start' and 'end' keys
            
        Returns:
            Dictionary with chart data
        """
        # Load log data
        log_data = self.data_access.load_well_log_data(well_name)
        if log_data is None or log_data.empty:
            return {
                "success": False,
                "error": f"Could not load log data for well {well_name}"
            }
        
        # Filter by depth if specified
        if depth_range:
            start = depth_range.get('start')
            end = depth_range.get('end')
            depth_col = None
            for col in log_data.columns:
                if 'DEPTH' in col.upper():
                    depth_col = col
                    break
            
            if depth_col and start is not None:
                log_data = self.data_access.filter_log_by_depth(
                    log_data, start, end, depth_col
                )
        
        # Find available curves
        available_curves = []
        for curve in curves:
            curve_variations = [curve, curve.upper(), curve.lower(), curve.capitalize()]
            for var in curve_variations:
                if var in log_data.columns:
                    available_curves.append(var)
                    break
        
        if not available_curves:
            return {
                "success": False,
                "error": f"None of the requested curves {curves} are available"
            }
        
        # Create chart
        depth_col = None
        for col in log_data.columns:
            if 'DEPTH' in col.upper():
                depth_col = col
                break
        
        chart = depth_plot(log_data, available_curves, depth_col=depth_col, well_name=well_name)
        
        return {
            "success": True,
            "chart": chart,
            "well_name": well_name,
            "curves": available_curves
        }
    
    def get_relevant_curves(self, property: str) -> Dict[str, Any]:
        """
        Get suggested relevant curves for a property.
        
        Args:
            property: The property being queried
            
        Returns:
            Dictionary with suggested curves
        """
        property_lower = property.lower()
        
        curve_mapping = {
            'shale volume': ['GR', 'VSH'],
            'vsh': ['GR', 'VSH'],
            'porosity': ['PHIF', 'PORD', 'NPHI'],
            'phif': ['PHIF', 'PORD'],
            'permeability': ['KLOGH', 'PHIF'],
            'klogh': ['KLOGH', 'PHIF'],
            'water saturation': ['SW', 'RW', 'PHIF'],
            'sw': ['SW', 'RW', 'PHIF'],
            'saturation': ['SW', 'RW', 'PHIF']
        }
        
        suggested = None
        for key, curves in curve_mapping.items():
            if key in property_lower:
                suggested = curves
                break
        
        if not suggested:
            # Default curves
            suggested = ['GR', 'PHIF', 'VSH', 'SW']
        
        return {
            "success": True,
            "property": property,
            "suggested_curves": suggested,
            "description": f"For {property}, these curves are most relevant: {', '.join(suggested)}"
        }
    
    def list_all_formations(self) -> Dict[str, Any]:
        """
        Get a complete list of all unique formation names from all wells.
        
        Returns:
            Dictionary with list of all unique formation names
        """
        all_formations = set()
        
        for well_name, formations in self.formation_tops_data.items():
            for formation in formations:
                formation_name = formation.get('formation_name', '')
                if formation_name:
                    all_formations.add(formation_name)
        
        sorted_formations = sorted(list(all_formations))
        
        return {
            "success": True,
            "formations": sorted_formations,
            "count": len(sorted_formations),
            "description": f"Found {len(sorted_formations)} unique formations across all wells"
        }
    
    def list_formations_by_well(self, well_name: str) -> Dict[str, Any]:
        """
        Get all formations for a specific well.
        
        Args:
            well_name: Name of the well
            
        Returns:
            Dictionary with formations for the well
        """
        # Try exact match first
        well_formations = None
        matched_well_name = None
        
        if well_name in self.formation_tops_data:
            well_formations = self.formation_tops_data[well_name]
            matched_well_name = well_name
        else:
            # Try partial match (well name might be formatted differently)
            well_name_upper = well_name.upper()
            for key, formations in self.formation_tops_data.items():
                if well_name_upper in key.upper() or key.upper() in well_name_upper:
                    well_formations = formations
                    matched_well_name = key
                    break
        
        if not well_formations:
            return {
                "success": False,
                "error": f"Could not find formations for well {well_name}",
                "well_name": well_name
            }
        
        formation_names = [f.get('formation_name', '') for f in well_formations if f.get('formation_name')]
        
        return {
            "success": True,
            "well_name": matched_well_name,
            "formations": formation_names,
            "count": len(formation_names),
            "description": f"Found {len(formation_names)} formations in well {matched_well_name}"
        }
    
    def list_all_wells_with_formations(self) -> Dict[str, Any]:
        """
        Get a complete mapping of all wells to their formations.
        
        Returns:
            Dictionary mapping well names to lists of formation names
        """
        wells_formations = {}
        
        for well_name, formations in self.formation_tops_data.items():
            formation_names = [f.get('formation_name', '') for f in formations if f.get('formation_name')]
            if formation_names:
                wells_formations[well_name] = formation_names
        
        return {
            "success": True,
            "wells_formations": wells_formations,
            "well_count": len(wells_formations),
            "total_formations": sum(len(f) for f in wells_formations.values()),
            "description": f"Found formations for {len(wells_formations)} wells"
        }
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> Dict[str, Any]:
        """
        Execute a tool by name with parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name == "plot_formation_log":
            return self.plot_formation_log(
                parameters.get('well_name'),
                parameters.get('formation_name'),
                parameters.get('curves', [])
            )
        elif tool_name == "get_formation_interval":
            return self.get_formation_interval(
                parameters.get('well_name'),
                parameters.get('formation_name')
            )
        elif tool_name == "calculate_formation_statistics":
            return self.calculate_formation_statistics(
                parameters.get('well_name'),
                parameters.get('formation_name'),
                parameters.get('curve')
            )
        elif tool_name == "plot_log_curves":
            return self.plot_log_curves(
                parameters.get('well_name'),
                parameters.get('curves', []),
                parameters.get('depth_range')
            )
        elif tool_name == "get_relevant_curves":
            return self.get_relevant_curves(parameters.get('property', ''))
        elif tool_name == "list_all_formations":
            return self.list_all_formations()
        elif tool_name == "list_formations_by_well":
            return self.list_formations_by_well(parameters.get('well_name', ''))
        elif tool_name == "list_all_wells_with_formations":
            return self.list_all_wells_with_formations()
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }


if __name__ == "__main__":
    # Test the tools
    from src.data_access import DataAccess
    
    access = DataAccess()
    tools = ComputationalTools(access)
    
    # Test tool schemas
    schemas = tools.get_tool_schemas()
    print(f"Available tools: {[s['name'] for s in schemas]}")
    
    # Test formation interval
    result = tools.get_formation_interval("15/9-F-1", "Hugin")
    print(f"\nFormation interval result: {result}")

