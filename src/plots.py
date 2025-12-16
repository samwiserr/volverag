"""
Altair plotting functions for well log data.
Following patterns from Geolog-Python-Loglan repository.
"""

import pandas as pd
import altair as alt
import numpy as np
from typing import List, Optional, Union, Dict
from pathlib import Path

# Configure Altair for better display
alt.data_transformers.enable('default', max_rows=None)


def depth_plot(df: pd.DataFrame, curves: List[str], depth_col: str = "DEPTH", 
                well_name: str = "", width: int = 200, height: int = 800) -> alt.Chart:
    """
    Create a depth plot showing multiple curves vs depth.
    
    Args:
        df: DataFrame with log data (must include depth column)
        curves: List of curve names to plot
        depth_col: Name of the depth column (default: "DEPTH")
        well_name: Name of the well (for title)
        width: Width of each curve panel
        height: Height of the plot
    
    Returns:
        Altair chart object
    """
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
    
    if depth_col not in df.columns:
        raise ValueError(f"Depth column '{depth_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Filter to available curves
    plot_curves = [c for c in curves if c in df.columns]
    if not plot_curves:
        return alt.Chart(pd.DataFrame()).mark_text(text="No curves available")
    
    # Prepare data for plotting
    plot_data = df[[depth_col] + plot_curves].copy()
    plot_data = plot_data.melt(
        id_vars=[depth_col],
        value_vars=plot_curves,
        var_name='curve',
        value_name='value'
    )
    
    # Remove null values for cleaner plots
    plot_data = plot_data.dropna(subset=['value'])
    
    # Create base chart
    base = alt.Chart(plot_data).mark_line(point=False, strokeWidth=1.5).encode(
        x=alt.X('value:Q', title='Value'),
        y=alt.Y(f'{depth_col}:Q', title='Depth (m)', sort='descending'),
        color=alt.Color('curve:N', title='Curve', scale=alt.Scale(scheme='category10')),
        tooltip=[depth_col, 'curve', 'value']
    )
    
    # Facet by curve
    title = f"{well_name} - Depth Plot" if well_name else "Depth Plot"
    chart = base.facet(
        column=alt.Column('curve:N', header=alt.Header(title=None))
    ).properties(
        title=title,
        width=width,
        height=height
    ).resolve_scale(y='shared')
    
    return chart


def crossplot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
              well_name: str = "", width: int = 400, height: int = 400) -> alt.Chart:
    """
    Create a crossplot (scatter plot) of two curves.
    
    Args:
        df: DataFrame with log data
        x: Name of curve for x-axis
        y: Name of curve for y-axis
        color: Optional name of curve for color encoding
        well_name: Name of the well (for title)
        width: Width of the plot
        height: Height of the plot
    
    Returns:
        Altair chart object
    """
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
    
    if x not in df.columns or y not in df.columns:
        missing = [c for c in [x, y] if c not in df.columns]
        raise ValueError(f"Curve(s) not found in DataFrame: {missing}. Available columns: {list(df.columns)}")
    
    # Prepare data
    plot_data = df[[x, y]].copy()
    if color and color in df.columns:
        plot_data[color] = df[color]
    
    # Remove null values
    plot_data = plot_data.dropna()
    
    if len(plot_data) == 0:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available after removing nulls")
    
    # Build encoding
    encoding = {
        'x': alt.X(f'{x}:Q', title=x),
        'y': alt.Y(f'{y}:Q', title=y),
        'tooltip': [x, y]
    }
    
    if color:
        encoding['color'] = alt.Color(f'{color}:Q', title=color, scale=alt.Scale(scheme='viridis'))
        encoding['tooltip'].append(color)
    
    # Create chart
    title = f"{well_name} - {x} vs {y}" if well_name else f"{x} vs {y}"
    chart = alt.Chart(plot_data).mark_circle(size=30, opacity=0.6).encode(**encoding).properties(
        title=title,
        width=width,
        height=height
    )
    
    return chart


def formation_depth_plot(df: pd.DataFrame, curves: List[str], formation_top: float,
                         formation_base: Optional[float], depth_col: str = "DEPTH",
                         well_name: str = "", formation_name: str = "",
                         width: int = 200, height: int = 800) -> alt.Chart:
    """
    Create a depth plot with formation interval highlighted.
    
    Args:
        df: DataFrame with log data
        curves: List of curve names to plot
        formation_top: Formation top depth (MD) in meters
        formation_base: Formation base depth (MD) in meters, or None to use max depth
        depth_col: Name of the depth column
        well_name: Name of the well
        formation_name: Name of the formation
        width: Width of each curve panel
        height: Height of the plot
    
    Returns:
        Altair chart with formation highlighted
    """
    if df.empty:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available")
    
    if depth_col not in df.columns:
        raise ValueError(f"Depth column '{depth_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Filter to available curves
    plot_curves = [c for c in curves if c in df.columns]
    if not plot_curves:
        return alt.Chart(pd.DataFrame()).mark_text(text="No curves available")
    
    # Prepare data for plotting
    plot_data = df[[depth_col] + plot_curves].copy()
    plot_data = plot_data.melt(
        id_vars=[depth_col],
        value_vars=plot_curves,
        var_name='curve',
        value_name='value'
    )
    
    # Remove null values
    plot_data = plot_data.dropna(subset=['value'])
    
    # Determine formation bounds
    depth_min = df[depth_col].min()
    depth_max = df[depth_col].max()
    formation_end = formation_base if formation_base is not None else depth_max
    
    # Create formation highlight data
    formation_data = pd.DataFrame({
        'formation_start': [formation_top],
        'formation_end': [formation_end],
        'formation_name': [formation_name or 'Formation']
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
    
    # Combine charts and facet
    title = f"{well_name} - {formation_name} Formation" if well_name and formation_name else "Formation Highlighted"
    chart = (highlight + base).facet(
        column=alt.Column('curve:N', header=alt.Header(title=None))
    ).properties(
        title=title,
        width=width,
        height=height
    ).resolve_scale(y='shared')
    
    return chart


def multi_well_comparison(well_data: List[Dict], curve: str, depth_top: float,
                          depth_base: float, depth_col: str = "DEPTH") -> alt.Chart:
    """
    Create a comparison plot showing the same curve from multiple wells.
    
    Args:
        well_data: List of dicts with keys 'well_name', 'df' (DataFrame), 'color' (optional)
        curve: Name of curve to compare
        depth_top: Top depth for comparison
        depth_base: Base depth for comparison
        depth_col: Name of depth column
    
    Returns:
        Altair chart with multiple wells overlaid
    """
    if not well_data:
        return alt.Chart(pd.DataFrame()).mark_text(text="No well data provided")
    
    # Prepare data from all wells
    all_data = []
    colors = alt.Scale(scheme='category10')
    
    for i, well_info in enumerate(well_data):
        df = well_info['df']
        well_name = well_info.get('well_name', f'Well {i+1}')
        color = well_info.get('color', colors.domain[i % len(colors.domain)])
        
        if depth_col not in df.columns or curve not in df.columns:
            continue
        
        # Filter by depth
        mask = (df[depth_col] >= depth_top) & (df[depth_col] <= depth_base)
        interval_df = df[mask].copy()
        
        if interval_df.empty:
            continue
        
        # Add well name
        interval_df['well_name'] = well_name
        all_data.append(interval_df[[depth_col, curve, 'well_name']])
    
    if not all_data:
        return alt.Chart(pd.DataFrame()).mark_text(text="No data available for comparison")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.rename(columns={curve: 'value'})
    combined_data = combined_data.dropna(subset=['value'])
    
    # Create chart
    chart = alt.Chart(combined_data).mark_line(point=False, strokeWidth=2).encode(
        x=alt.X('value:Q', title=curve),
        y=alt.Y(f'{depth_col}:Q', title='Depth (m)', sort='descending'),
        color=alt.Color('well_name:N', title='Well', scale=alt.Scale(scheme='category10')),
        tooltip=[depth_col, 'value', 'well_name']
    ).properties(
        title=f"{curve} Comparison Across Wells",
        width=400,
        height=600
    )
    
    return chart


if __name__ == "__main__":
    # Test the plots module
    print("Testing plots module...")
    
    # Create sample data
    depth = np.arange(2600, 2800, 0.15)
    sample_df = pd.DataFrame({
        'DEPTH': depth,
        'GR': 50 + 20 * np.sin(depth / 10) + np.random.normal(0, 5, len(depth)),
        'RHOB': 2.3 + 0.1 * np.cos(depth / 15) + np.random.normal(0, 0.05, len(depth)),
        'NPHI': 0.2 + 0.05 * np.sin(depth / 12) + np.random.normal(0, 0.01, len(depth))
    })
    
    print("Creating depth plot...")
    chart1 = depth_plot(sample_df, ['GR', 'RHOB', 'NPHI'], well_name="Test Well")
    print("✓ Depth plot created")
    
    print("Creating crossplot...")
    chart2 = crossplot(sample_df, 'RHOB', 'NPHI', color='GR', well_name="Test Well")
    print("✓ Crossplot created")
    
    print("Creating formation plot...")
    chart3 = formation_depth_plot(
        sample_df, ['GR', 'RHOB'], 2650, 2750,
        well_name="Test Well", formation_name="Test Formation"
    )
    print("✓ Formation plot created")
    
    print("\nAll plot functions working correctly!")

