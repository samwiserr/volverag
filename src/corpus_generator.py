"""
Document corpus generator for RAG system.
Generates textual documents from LAS files for semantic search.
"""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from src.las_io import load_las, get_interval_df, get_well_metadata, list_curves


def generate_header_document(las, file_path: Path) -> Dict:
    """
    Generate a markdown document from LAS file header metadata.
    
    Args:
        las: LASFile object
        file_path: Path to the LAS file
    
    Returns:
        Dictionary with 'text' and 'metadata' keys
    """
    metadata = get_well_metadata(las)
    curves = list_curves(las, include_stats=False)
    
    # Build markdown document
    well_name = metadata.get('well_name', file_path.parent.name)
    field = metadata.get('field', 'Volve')
    location = metadata.get('location', '')
    uwi = metadata.get('uwi', '')
    company = metadata.get('company', '')
    service_company = metadata.get('service_company', '')
    start_depth = metadata.get('start_depth')
    stop_depth = metadata.get('stop_depth')
    step = metadata.get('step')
    date = metadata.get('date', '')
    kb = metadata.get('kb')
    df = metadata.get('df')
    
    # Build document text
    doc_lines = [f"# Well: {well_name} ({field})"]
    doc_lines.append("")
    
    if location:
        doc_lines.append(f"**Location:** {location}")
    if uwi:
        doc_lines.append(f"**UWI:** {uwi}")
    if company:
        doc_lines.append(f"**Operator:** {company}")
    if service_company:
        doc_lines.append(f"**Logging Company:** {service_company}")
    
    doc_lines.append("")
    
    if start_depth is not None and stop_depth is not None:
        depth_range = f"{start_depth:.1f}–{stop_depth:.1f} m MD"
        if step:
            depth_range += f", step {step:.2f} m"
        doc_lines.append(f"**Depth range:** {depth_range}")
    
    if kb is not None:
        doc_lines.append(f"**KB (Kelly Bushing):** {kb:.2f} m")
    if df is not None:
        doc_lines.append(f"**DF (Datum Elevation):** {df:.2f} m")
    
    if date:
        doc_lines.append(f"**Logging Date:** {date}")
    
    doc_lines.append("")
    doc_lines.append("**Available Curves:**")
    curve_names = [c.mnemonic for c in curves]
    if curve_names:
        # Group curves by type
        common_curves = ['GR', 'RHOB', 'NPHI', 'DT', 'RT', 'CALI', 'PHIF', 'SW', 'VSH', 'KLOGH']
        other_curves = [c for c in curve_names if c not in common_curves]
        
        doc_lines.append(f"- Primary curves: {', '.join([c for c in curve_names if c in common_curves])}")
        if other_curves:
            doc_lines.append(f"- Additional curves: {', '.join(other_curves[:20])}")  # Limit to first 20
            if len(other_curves) > 20:
                doc_lines.append(f"- ... and {len(other_curves) - 20} more curves")
    else:
        doc_lines.append("- No curves listed")
    
    doc_lines.append("")
    doc_lines.append("**File Information:**")
    doc_lines.append(f"- File: {file_path.name}")
    doc_lines.append(f"- Folder: {file_path.parent.name}")
    
    # Add any notable notes from header
    if hasattr(las, 'other') and las.other:
        notes = []
        for item in las.other:
            if hasattr(item, 'descr') and item.descr:
                notes.append(str(item.descr))
        if notes:
            doc_lines.append("")
            doc_lines.append("**Notes:**")
            for note in notes[:5]:  # Limit to first 5 notes
                doc_lines.append(f"- {note}")
    
    doc_text = "\n".join(doc_lines)
    
    # Build metadata
    doc_metadata = {
        'well_name': well_name,
        'field': field,
        'doc_type': 'header',
        'file_path': str(file_path),
        'file_name': file_path.name,
        'well_folder': file_path.parent.name,
        'curve_list': curve_names,
        'depth_top': start_depth,
        'depth_base': stop_depth,
        'date': date,
        'company': company,
        'service_company': service_company
    }
    
    return {
        'text': doc_text,
        'metadata': doc_metadata
    }


def generate_interval_summary(las, depth_top: float, depth_base: float, 
                             well_name: str, formation_name: Optional[str] = None) -> str:
    """
    Generate a templated summary for a depth interval.
    
    Args:
        las: LASFile object
        depth_top: Top depth (MD) in meters
        depth_base: Base depth (MD) in meters
        well_name: Name of the well
        formation_name: Optional formation name
    
    Returns:
        Summary text string
    """
    try:
        # Get interval data
        interval_df = get_interval_df(las, depth_top, depth_base)
        
        if interval_df.empty:
            return f"In well {well_name} from {depth_top:.1f}–{depth_base:.1f} m MD, no data is available."
        
        # Find depth column
        depth_col = None
        for col in interval_df.columns:
            if 'DEPTH' in col.upper() or col.upper() == 'MD':
                depth_col = col
                break
        
        if depth_col is None:
            depth_col = interval_df.columns[0]
        
        # Compute statistics for key curves
        key_curves = ['GR', 'RHOB', 'NPHI', 'DT', 'RT', 'PHIF', 'SW', 'VSH', 'KLOGH']
        available_curves = [c for c in key_curves if c in interval_df.columns]
        
        if not available_curves:
            # Use first few numeric columns
            numeric_cols = interval_df.select_dtypes(include=[np.number]).columns.tolist()
            if depth_col in numeric_cols:
                numeric_cols.remove(depth_col)
            available_curves = numeric_cols[:5]  # Limit to first 5
        
        # Build summary text
        summary_parts = []
        
        if formation_name:
            summary_parts.append(f"In well {well_name}, the {formation_name} formation ({depth_top:.1f}–{depth_base:.1f} m MD)")
        else:
            summary_parts.append(f"In well {well_name} from {depth_top:.1f}–{depth_base:.1f} m MD")
        
        summary_parts.append("exhibits the following petrophysical characteristics:")
        summary_parts.append("")
        
        # Add statistics for each curve
        for curve in available_curves[:8]:  # Limit to 8 curves
            curve_data = interval_df[curve].dropna()
            if len(curve_data) == 0:
                continue
            
            mean_val = curve_data.mean()
            min_val = curve_data.min()
            max_val = curve_data.max()
            std_val = curve_data.std()
            null_pct = (1 - len(curve_data) / len(interval_df)) * 100
            
            # Format based on curve type
            if curve == 'GR':
                summary_parts.append(f"- **Gamma Ray (GR):** ranges from {min_val:.1f}–{max_val:.1f} API (mean {mean_val:.1f} API, std {std_val:.1f} API), "
                                    f"indicating {'high' if mean_val > 80 else 'moderate' if mean_val > 50 else 'low'} shale content.")
            elif curve == 'RHOB':
                summary_parts.append(f"- **Bulk Density (RHOB):** ranges from {min_val:.2f}–{max_val:.2f} g/cc (mean {mean_val:.2f} g/cc).")
            elif curve == 'NPHI':
                summary_parts.append(f"- **Neutron Porosity (NPHI):** ranges from {min_val:.3f}–{max_val:.3f} v/v (mean {mean_val:.3f} v/v).")
            elif curve == 'PHIF':
                summary_parts.append(f"- **Effective Porosity (PHIF):** ranges from {min_val:.3f}–{max_val:.3f} v/v (mean {mean_val:.3f} v/v).")
            elif curve == 'SW':
                summary_parts.append(f"- **Water Saturation (SW):** ranges from {min_val:.3f}–{max_val:.3f} v/v (mean {mean_val:.3f} v/v).")
            elif curve == 'VSH':
                summary_parts.append(f"- **Volume of Shale (VSH):** ranges from {min_val:.3f}–{max_val:.3f} v/v (mean {mean_val:.3f} v/v).")
            elif curve == 'KLOGH':
                summary_parts.append(f"- **Permeability (KLOGH):** ranges from {min_val:.2f}–{max_val:.2f} mD (mean {mean_val:.2f} mD).")
            elif curve == 'DT':
                summary_parts.append(f"- **Sonic Travel Time (DT):** ranges from {min_val:.1f}–{max_val:.1f} μs/ft (mean {mean_val:.1f} μs/ft).")
            elif curve == 'RT':
                summary_parts.append(f"- **Resistivity (RT):** ranges from {min_val:.2f}–{max_val:.2f} Ω·m (mean {mean_val:.2f} Ω·m).")
            else:
                summary_parts.append(f"- **{curve}:** ranges from {min_val:.2f}–{max_val:.2f} (mean {mean_val:.2f}, std {std_val:.2f}).")
            
            if null_pct > 5:
                summary_parts[-1] = summary_parts[-1].rstrip('.') + f" ({null_pct:.1f}% missing data)."
            else:
                summary_parts[-1] = summary_parts[-1].rstrip('.') + "."
        
        summary_text = "\n".join(summary_parts)
        return summary_text
    
    except Exception as e:
        return f"In well {well_name} from {depth_top:.1f}–{depth_base:.1f} m MD, data extraction encountered an error: {str(e)}"


def generate_well_documents(well_folder: Path, formation_tops: Optional[Dict] = None) -> List[Dict]:
    """
    Generate all documents for a well folder.
    
    Args:
        well_folder: Path to well folder
        formation_tops: Optional dictionary mapping well_name to list of formation tops
    
    Returns:
        List of document dictionaries
    """
    documents = []
    
    # Find all LAS files
    las_files = list(well_folder.glob("*.las")) + list(well_folder.glob("*.LAS"))
    
    if not las_files:
        return documents
    
    # Get well name from folder
    well_name = well_folder.name
    
    # Get formation tops for this well if available
    well_formations = None
    if formation_tops and well_name in formation_tops:
        well_formations = formation_tops[well_name]
    
    for las_file in las_files:
        try:
            # Load LAS file
            las, df = load_las(las_file, return_dataframe='both')
            
            # Generate header document
            header_doc = generate_header_document(las, las_file)
            documents.append(header_doc)
            
            # Generate interval summaries
            # 1. Full well summary (if reasonable size)
            if df is not None and not df.empty:
                depth_col = None
                for col in df.columns:
                    if 'DEPTH' in col.upper() or col.upper() == 'MD':
                        depth_col = col
                        break
                
                if depth_col:
                    depth_min = df[depth_col].min()
                    depth_max = df[depth_col].max()
                    depth_range = depth_max - depth_min
                    
                    # Generate 2-3 interval summaries
                    if well_formations:
                        # Generate summaries for each formation
                        for formation in well_formations[:3]:  # Limit to first 3 formations
                            formation_name = formation.get('formation', 'Unknown')
                            formation_top = formation.get('md', depth_min)
                            formation_base = formation.get('md', depth_max)
                            
                            # If only one MD value, create interval around it
                            if formation_top == formation_base:
                                interval_size = min(50, depth_range / 4)
                                formation_top = max(depth_min, formation_top - interval_size / 2)
                                formation_base = min(depth_max, formation_base + interval_size / 2)
                            
                            summary_text = generate_interval_summary(
                                las, formation_top, formation_base, well_name, formation_name
                            )
                            
                            # Create document
                            interval_doc = {
                                'text': summary_text,
                                'metadata': {
                                    'well_name': well_name,
                                    'field': header_doc['metadata'].get('field', 'Volve'),
                                    'doc_type': 'interval_summary',
                                    'formation': formation_name,
                                    'depth_top': formation_top,
                                    'depth_base': formation_base,
                                    'file_path': str(las_file),
                                    'curve_list': header_doc['metadata'].get('curve_list', [])
                                }
                            }
                            documents.append(interval_doc)
                    else:
                        # Generate generic interval summaries
                        if depth_range > 100:
                            # Split into 2-3 intervals
                            n_intervals = min(3, int(depth_range / 200))
                            interval_size = depth_range / n_intervals
                            
                            for i in range(n_intervals):
                                interval_top = depth_min + i * interval_size
                                interval_base = depth_min + (i + 1) * interval_size
                                
                                summary_text = generate_interval_summary(
                                    las, interval_top, interval_base, well_name
                                )
                                
                                interval_doc = {
                                    'text': summary_text,
                                    'metadata': {
                                        'well_name': well_name,
                                        'field': header_doc['metadata'].get('field', 'Volve'),
                                        'doc_type': 'interval_summary',
                                        'depth_top': interval_top,
                                        'depth_base': interval_base,
                                        'file_path': str(las_file),
                                        'curve_list': header_doc['metadata'].get('curve_list', [])
                                    }
                                }
                                documents.append(interval_doc)
        
        except Exception as e:
            print(f"Error processing {las_file}: {e}")
            continue
    
    return documents


def generate_master_formations_document(formation_tops_data: Dict[str, List[Dict]]) -> Dict:
    """
    Generate a master document listing all formations from all wells.
    This document serves as a complete reference for "list all formations" queries.
    
    Args:
        formation_tops_data: Dictionary mapping well names to lists of formation tops
        
    Returns:
        Dictionary with 'text' and 'metadata' keys
    """
    doc_lines = ["# Global Formations in Volve"]
    doc_lines.append("")
    doc_lines.append("Complete list of all formations from all wells and marker files:")
    doc_lines.append("")
    
    # Group formations by well
    for well_name in sorted(formation_tops_data.keys()):
        formations = formation_tops_data[well_name]
        if not formations:
            continue
        
        doc_lines.append(f"## Well: {well_name}")
        doc_lines.append("")
        
        # List all formations for this well
        for formation in formations:
            formation_name = formation.get('formation_name', '')
            if not formation_name:
                continue
            
            md = formation.get('md')
            tvd = formation.get('tvd')
            tvdss = formation.get('tvdss')
            
            formation_line = f"- {formation_name}"
            if md is not None:
                formation_line += f" (MD: {md:.1f}m"
                if tvd is not None:
                    formation_line += f", TVD: {tvd:.1f}m"
                if tvdss is not None:
                    formation_line += f", TVDSS: {tvdss:.1f}m"
                formation_line += ")"
            
            doc_lines.append(formation_line)
        
        doc_lines.append("")
    
    # Add summary statistics
    total_wells = len([w for w in formation_tops_data.keys() if formation_tops_data[w]])
    all_formations = set()
    for formations in formation_tops_data.values():
        for f in formations:
            formation_name = f.get('formation_name', '')
            if formation_name:
                all_formations.add(formation_name)
    
    doc_lines.append("---")
    doc_lines.append("")
    doc_lines.append(f"**Summary:** {total_wells} wells, {len(all_formations)} unique formations")
    doc_lines.append("")
    doc_lines.append("**Unique Formations:**")
    for formation_name in sorted(all_formations):
        doc_lines.append(f"- {formation_name}")
    
    doc_text = "\n".join(doc_lines)
    
    # Build metadata
    doc_metadata = {
        'doc_type': 'global_formations',
        'type': 'master_list',
        'well_count': total_wells,
        'formation_count': len(all_formations),
        'description': 'Complete list of all formations from all wells in the Volve dataset'
    }
    
    return {
        'text': doc_text,
        'metadata': doc_metadata
    }


if __name__ == "__main__":
    # Test the corpus generator
    test_folder = Path("spwla_volve-main/15_9-F-1")
    
    if test_folder.exists():
        print("Testing corpus generator...")
        
        # Load a test LAS file
        las_files = list(test_folder.glob("*.LAS"))
        if las_files:
            las, df = load_las(las_files[0], return_dataframe='both')
            
            # Test header document
            header_doc = generate_header_document(las, las_files[0])
            print("\n=== Header Document ===")
            print(header_doc['text'][:500] + "...")
            print(f"\nMetadata: {list(header_doc['metadata'].keys())}")
            
            # Test interval summary
            if df is not None and not df.empty:
                depth_col = None
                for col in df.columns:
                    if 'DEPTH' in col.upper():
                        depth_col = col
                        break
                
                if depth_col:
                    depth_min = df[depth_col].min()
                    depth_max = df[depth_col].max()
                    interval_top = depth_min
                    interval_base = min(depth_min + 100, depth_max)
                    
                    summary = generate_interval_summary(las, interval_top, interval_base, "Test Well")
                    print("\n=== Interval Summary ===")
                    print(summary)
        
        # Test well documents
        print("\n=== Generating Well Documents ===")
        well_docs = generate_well_documents(test_folder)
        print(f"Generated {len(well_docs)} documents")
        for i, doc in enumerate(well_docs[:3]):
            print(f"\nDocument {i+1} ({doc['metadata']['doc_type']}): {len(doc['text'])} chars")
    else:
        print(f"Test folder not found: {test_folder}")

