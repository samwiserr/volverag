"""
Visualization module for well log data using Altair.
Following patterns from the referenced Geolog-Python-Loglan notebook.
Enhanced with formation highlighting capabilities.
"""

import pandas as pd
import altair as alt
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class WellLogVisualizer:
    """Creates visualizations for well log data using Altair."""
    
    def __init__(self):
        """Initialize the visualizer."""
        # Configure Altair for better display
        alt.data_transformers.enable('default', max_rows=None)
    
    def plot_formation_with_highlight(self, well_name: str, formation_name: str,
                                     curves: List[str], log_data: pd.DataFrame,
                                     formation_top: float, formation_base: Optional[float],
                                     depth_col: str = "DEPTH") -> alt.Chart:
        """
        Create a log plot with formation interval highlighted.
        
        Args:
            well_name: Name of the well
            formation_name: Name of the formation
            curves: List of curves to plot
            log_data: DataFrame with log data
            formation_top: Formation top depth (MD)
            formation_base: Formation base depth (MD), or None
            depth_col: Name of depth column
            
        Returns:
            Altair chart with formation highlighted
        """
        if log_data.empty or depth_col not in log_data.columns:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
        
        # Prepare data for plotting
        plot_curves = [c for c in curves if c in log_data.columns]
        if not plot_curves:
            return alt.Chart(pd.DataFrame()).mark_text(text="No curves available")
        
        plot_data = log_data[[depth_col] + plot_curves].copy()
        plot_data = plot_data.melt(id_vars=[depth_col],
                                 value_vars=plot_curves,
                                 var_name='curve', value_name='value')
        
        # Create formation highlight data
        depth_min = log_data[depth_col].min()
        depth_max = log_data[depth_col].max()
        formation_end = formation_base if formation_base else depth_max
        
        formation_data = pd.DataFrame({
            'formation_start': [formation_top],
            'formation_end': [formation_end],
            'formation_name': [formation_name]
        })
        
        # Create base chart for curves
        base = alt.Chart(plot_data).mark_line(point=False, strokeWidth=1.5).encode(
            x=alt.X('value:Q', title='Value'),
            y=alt.Y(f'{depth_col}:Q', title='Depth (m)', sort='descending'),
            color=alt.Color('curve:N', title='Curve', scale=alt.Scale(scheme='category10')),
            tooltip=[depth_col, 'curve', 'value']
        )
        
        # Create formation highlight layer
        highlight = alt.Chart(formation_data).mark_rect(
            opacity=0.2,
            color='yellow'
        ).encode(
            y=alt.Y('formation_start:Q', title='Depth (m)'),
            y2=alt.Y('formation_end:Q'),
            tooltip=[alt.Tooltip('formation_name:N', title='Formation')]
        )
        
        # Combine charts
        chart = (highlight + base).facet(
            column=alt.Column('curve:N', header=alt.Header(title=None))
        ).properties(
            title=f"{well_name} - {formation_name} Formation Highlighted",
            width=200,
            height=800
        ).resolve_scale(y='shared')
        
        return chart
    
    def create_log_plot(self, log_data: pd.DataFrame, curves: List[str], 
                       well_name: str = "", depth_col: str = "DEPTH") -> alt.Chart:
        """
        Create a log plot showing multiple curves vs depth.
        
        Args:
            log_data: DataFrame with log data
            curves: List of curve names to plot
            well_name: Name of the well
            depth_col: Name of the depth column
            
        Returns:
            Altair chart object
        """
        if log_data.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
        
        # Prepare data for plotting
        plot_data = log_data[[depth_col] + [c for c in curves if c in log_data.columns]].copy()
        plot_data = plot_data.melt(id_vars=[depth_col], 
                                   value_vars=[c for c in curves if c in log_data.columns],
                                   var_name='curve', value_name='value')
        
        # Create base chart
        base = alt.Chart(plot_data).mark_line(point=False, strokeWidth=1).encode(
            x=alt.X('value:Q', title='Value'),
            y=alt.Y(f'{depth_col}:Q', title='Depth (m)', sort='descending'),
            color=alt.Color('curve:N', title='Curve'),
            tooltip=[depth_col, 'curve', 'value']
        )
        
        # Facet by curve
        chart = base.facet(
            column=alt.Column('curve:N', header=alt.Header(title=None))
        ).properties(
            title=f"{well_name} - Well Logs" if well_name else "Well Logs",
            width=150,
            height=600
        )
        
        return chart
    
    def create_porosity_permeability_plot(self, log_data: pd.DataFrame,
                                         well_name: str = "",
                                         phif_col: str = "PHIF",
                                         klogh_col: str = "KLOGH") -> alt.Chart:
        """
        Create a porosity vs permeability crossplot.
        
        Args:
            log_data: DataFrame with log data
            well_name: Name of the well
            phif_col: Name of porosity column
            klogh_col: Name of permeability column
            
        Returns:
            Altair chart object
        """
        if log_data.empty or phif_col not in log_data.columns or klogh_col not in log_data.columns:
            return alt.Chart(pd.DataFrame()).mark_text(text="Data not available")
        
        # Filter out null values
        plot_data = log_data[[phif_col, klogh_col]].dropna()
        
        if plot_data.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No valid data points")
        
        chart = alt.Chart(plot_data).mark_circle(size=30, opacity=0.6).encode(
            x=alt.X(f'{phif_col}:Q', title='Porosity (PHIF)', scale=alt.Scale(zero=False)),
            y=alt.Y(f'{klogh_col}:Q', title='Permeability (KLOGH, mD)', 
                   scale=alt.Scale(type='log', zero=False)),
            tooltip=[phif_col, klogh_col]
        ).properties(
            title=f"{well_name} - Porosity vs Permeability" if well_name else "Porosity vs Permeability",
            width=400,
            height=300
        )
        
        return chart
    
    def create_saturation_plot(self, log_data: pd.DataFrame,
                              well_name: str = "",
                              sw_col: str = "SW",
                              depth_col: str = "DEPTH") -> alt.Chart:
        """
        Create a water saturation vs depth plot.
        
        Args:
            log_data: DataFrame with log data
            well_name: Name of the well
            sw_col: Name of water saturation column
            depth_col: Name of depth column
            
        Returns:
            Altair chart object
        """
        if log_data.empty or sw_col not in log_data.columns:
            return alt.Chart(pd.DataFrame()).mark_text(text="Data not available")
        
        plot_data = log_data[[depth_col, sw_col]].dropna()
        
        if plot_data.empty:
            return alt.Chart(pd.DataFrame()).mark_text(text="No valid data points")
        
        chart = alt.Chart(plot_data).mark_line(point=False, strokeWidth=2).encode(
            x=alt.X(f'{sw_col}:Q', title='Water Saturation (SW)', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f'{depth_col}:Q', title='Depth (m)', sort='descending'),
            tooltip=[depth_col, sw_col]
        ).properties(
            title=f"{well_name} - Water Saturation" if well_name else "Water Saturation",
            width=200,
            height=600
        )
        
        return chart
    
    def create_multi_well_comparison(self, well_data_list: List[Dict],
                                    curve: str = "PHIF") -> alt.Chart:
        """
        Create a comparison plot for multiple wells.
        
        Args:
            well_data_list: List of dictionaries with 'well_name' and 'log_data' keys
            curve: Curve name to compare
            
        Returns:
            Altair chart object
        """
        # Combine data from all wells
        combined_data = []
        
        for well_data in well_data_list:
            well_name = well_data.get('well_name', 'Unknown')
            log_data = well_data.get('log_data', pd.DataFrame())
            
            if not log_data.empty and curve in log_data.columns:
                # Find depth column
                depth_col = None
                for col in log_data.columns:
                    if 'DEPTH' in col.upper():
                        depth_col = col
                        break
                
                if depth_col:
                    plot_df = log_data[[depth_col, curve]].copy()
                    plot_df['well_name'] = well_name
                    plot_df = plot_df.dropna()
                    combined_data.append(plot_df)
        
        if not combined_data:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        chart = alt.Chart(combined_df).mark_line(point=False, strokeWidth=1.5).encode(
            x=alt.X(f'{curve}:Q', title=curve),
            y=alt.Y('DEPTH:Q', title='Depth (m)', sort='descending'),
            color=alt.Color('well_name:N', title='Well'),
            tooltip=['well_name', 'DEPTH', curve]
        ).properties(
            title=f"Multi-Well Comparison - {curve}",
            width=300,
            height=600
        )
        
        return chart
    
    def create_statistics_summary(self, well_summaries: List[Dict]) -> alt.Chart:
        """
        Create a summary chart showing statistics across wells.
        
        Args:
            well_summaries: List of well summary dictionaries
            
        Returns:
            Altair chart object
        """
        # Extract statistics for plotting
        stats_data = []
        
        for summary in well_summaries:
            well_name = summary.get('well_name', 'Unknown')
            curve_stats = summary.get('curve_statistics', {})
            
            for curve, stats in curve_stats.items():
                if stats and 'mean' in stats:
                    stats_data.append({
                        'well_name': well_name,
                        'curve': curve,
                        'mean': stats['mean'],
                        'min': stats['min'],
                        'max': stats['max']
                    })
        
        if not stats_data:
            return alt.Chart(pd.DataFrame()).mark_text(text="No statistics available")
        
        df = pd.DataFrame(stats_data)
        
        # Create bar chart
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('well_name:N', title='Well'),
            y=alt.Y('mean:Q', title='Mean Value'),
            color=alt.Color('curve:N', title='Curve'),
            tooltip=['well_name', 'curve', 'mean', 'min', 'max']
        ).properties(
            title="Well Statistics Summary",
            width=600,
            height=400
        )
        
        return chart
    
    def plot_formation_with_highlight(self, well_name: str, formation_name: str,
                                     curves: List[str], log_data: pd.DataFrame,
                                     formation_top: float, formation_base: Optional[float],
                                     depth_col: str = "DEPTH") -> alt.Chart:
        """
        Create a log plot with formation interval highlighted.
        
        Args:
            well_name: Name of the well
            formation_name: Name of the formation
            curves: List of curves to plot
            log_data: DataFrame with log data
            formation_top: Formation top depth (MD)
            formation_base: Formation base depth (MD), or None
            depth_col: Name of depth column
            
        Returns:
            Altair chart with formation highlighted
        """
        if log_data.empty or depth_col not in log_data.columns:
            return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
        
        # Prepare data for plotting
        plot_curves = [c for c in curves if c in log_data.columns]
        if not plot_curves:
            return alt.Chart(pd.DataFrame()).mark_text(text="No curves available")
        
        plot_data = log_data[[depth_col] + plot_curves].copy()
        plot_data = plot_data.melt(id_vars=[depth_col],
                                 value_vars=plot_curves,
                                 var_name='curve', value_name='value')
        
        # Create formation highlight data
        depth_min = log_data[depth_col].min()
        depth_max = log_data[depth_col].max()
        formation_end = formation_base if formation_base else depth_max
        
        formation_data = pd.DataFrame({
            'formation_start': [formation_top],
            'formation_end': [formation_end],
            'formation_name': [formation_name]
        })
        
        # Create base chart for curves
        base = alt.Chart(plot_data).mark_line(point=False, strokeWidth=1.5).encode(
            x=alt.X('value:Q', title='Value'),
            y=alt.Y(f'{depth_col}:Q', title='Depth (m)', sort='descending'),
            color=alt.Color('curve:N', title='Curve', scale=alt.Scale(scheme='category10')),
            tooltip=[depth_col, 'curve', 'value']
        )
        
        # Create formation highlight layer
        highlight = alt.Chart(formation_data).mark_rect(
            opacity=0.2,
            color='yellow'
        ).encode(
            y=alt.Y('formation_start:Q', title='Depth (m)'),
            y2=alt.Y('formation_end:Q'),
            tooltip=[alt.Tooltip('formation_name:N', title='Formation')]
        )
        
        # Combine charts - overlay highlight on each curve panel
        chart = (highlight + base).facet(
            column=alt.Column('curve:N', header=alt.Header(title=None))
        ).properties(
            title=f"{well_name} - {formation_name} Formation Highlighted",
            width=200,
            height=800
        ).resolve_scale(y='shared')
        
        return chart
    
    def get_relevant_curves_for_query(self, query: str, curve_mentioned: Optional[str] = None) -> List[str]:
        """
        Determine relevant curves to plot based on query.
        
        Args:
            query: User query text
            curve_mentioned: Explicitly mentioned curve name
            
        Returns:
            List of relevant curve names
        """
        query_lower = query.lower()
        
        # If curve explicitly mentioned, prioritize it
        if curve_mentioned:
            curves = [curve_mentioned.upper()]
        else:
            curves = []
        
        # Map query keywords to relevant curves
        if 'shale' in query_lower or 'vsh' in query_lower:
            if 'GR' not in curves:
                curves.insert(0, 'GR')  # Gamma Ray is primary for shale
            if 'VSH' not in curves:
                curves.append('VSH')
        elif 'porosity' in query_lower or 'phif' in query_lower:
            if 'PHIF' not in curves:
                curves.insert(0, 'PHIF')
            if 'PORD' not in curves:
                curves.append('PORD')
        elif 'permeability' in query_lower or 'klogh' in query_lower:
            if 'KLOGH' not in curves:
                curves.insert(0, 'KLOGH')
            if 'PHIF' not in curves:
                curves.append('PHIF')  # Often plotted together
        elif 'saturation' in query_lower or 'sw' in query_lower:
            if 'SW' not in curves:
                curves.insert(0, 'SW')
            if 'RW' not in curves:
                curves.append('RW')
        
        # Default curves if nothing specific
        if not curves:
            curves = ['GR', 'PHIF', 'VSH']
        
        return curves
    
    def save_chart(self, chart: alt.Chart, filepath: str, format: str = "html") -> None:
        """
        Save chart to file.
        
        Args:
            chart: Altair chart object
            filepath: Path to save file
            format: File format ("html", "png", "svg", "json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "html":
            chart.save(str(filepath))
        elif format == "png":
            chart.save(str(filepath), scale_factor=2)
        elif format == "svg":
            chart.save(str(filepath))
        elif format == "json":
            chart.save(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Chart saved to {filepath}")


if __name__ == "__main__":
    # Test the visualizer
    import numpy as np
    
    # Create sample data
    depth = np.arange(3000, 3500, 0.1)
    phif = 0.15 + 0.05 * np.sin(depth / 50) + np.random.normal(0, 0.01, len(depth))
    klogh = 10 ** (2 * phif - 1) + np.random.normal(0, 5, len(depth))
    sw = 0.3 + 0.2 * np.sin(depth / 30) + np.random.normal(0, 0.05, len(depth))
    
    sample_data = pd.DataFrame({
        'DEPTH': depth,
        'PHIF': phif,
        'KLOGH': klogh,
        'SW': sw
    })
    
    # Test visualizer
    viz = WellLogVisualizer()
    
    # Create plots
    log_plot = viz.create_log_plot(sample_data, ['PHIF', 'SW'], well_name="Test Well")
    poro_perm_plot = viz.create_porosity_permeability_plot(sample_data, well_name="Test Well")
    
    print("Visualization module initialized successfully")
    print("Sample charts created (not displayed in test mode)")

