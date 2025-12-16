"""Test the formation tops parser to verify header row detection."""

from src.formation_tops_parser import FormationTopsParser

parser = FormationTopsParser()
well_data = parser.parse_file()

print(f"Parsed formation tops for {len(well_data)} wells\n")

# Check for header rows in formation names
header_keywords = ['name', 'surface name', 'obs#', 'qlf', 'md', 'tvd', 'tvdss', 'twt', 'dip', 'azi', 'easting', 'northing', 'intrp']
header_rows_found = []

for well_name, formations in well_data.items():
    print(f"Well: {well_name}")
    print(f"  Formations: {len(formations)}")
    
    for formation in formations:
        formation_name = formation.get('formation_name', '')
        # Check if this looks like a header row
        if any(kw in formation_name.lower() for kw in ['name surface name', 'obs#', 'qlf md tvd']):
            header_rows_found.append(f"{well_name}: {formation_name}")
            print(f"  ⚠ Header row detected: {formation_name[:80]}")
        else:
            print(f"  ✓ {formation_name} (MD: {formation.get('md', 'N/A')})")
    
    print()

if header_rows_found:
    print(f"\n✗ Found {len(header_rows_found)} header rows:")
    for hr in header_rows_found:
        print(f"  - {hr}")
else:
    print("\n✓ No header rows found in formations")

