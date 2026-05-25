"""Tests for chapter-6 weighted-average petrophysical table parsing."""
import pytest

from src.tools.petro_params_tool import _parse_weighted_average_rows

pytestmark = pytest.mark.unit


SAMPLE_BLOCK = """
6 Petrophysical results
15/9-F-1 C Averages
Formation / Top Base N/G PHIF SW KLOGH
Weighted Heather averages: 0.036 0.123 0.896 0.3 0.2 0.3
Weighted Middle Hugin averages: 0.967 0.219 0.338 1677 11 299
Weighted Hugin 1.2 (2) averages: 0.972 0.217 0.383 314 103 208
"""


def test_parse_weighted_average_rows_extracts_formations():
    rows = _parse_weighted_average_rows(SAMPLE_BLOCK, "PETROPHYSICAL_REPORT_1.pdf", 21, 22)
    assert len(rows) == 3
    wells = {r.well for r in rows}
    assert any("15/9-F-1" in w for w in wells)
    hugin_rows = [r for r in rows if "hugin" in r.formation.lower()]
    assert len(hugin_rows) >= 2
    middle = next(r for r in rows if "Middle Hugin" in r.formation)
    assert middle.sw == 0.338
    assert middle.phif == 0.219


def test_parse_weighted_average_rows_uses_well_folder_in_path():
    rows = _parse_weighted_average_rows(
        "Weighted Hugin averages: 0.888 0.207 0.232 512 2 123",
        r"C:\data\spwla-volve-main\15_9-F-5\PETROPHYSICAL_REPORT_1.PDF",
        None,
        None,
    )
    assert len(rows) == 1
    assert rows[0].well == "15/9-F-5"
    assert rows[0].formation == "Hugin"
    assert rows[0].sw == 0.232
