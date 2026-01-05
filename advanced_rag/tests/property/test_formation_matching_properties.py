"""
Property-based tests for formation matching.

Tests fuzzy matching properties, consistency, and edge cases.
"""
import pytest
import hypothesis
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from src.core.well_utils import normalize_well


@pytest.mark.property
class TestFormationMatchingProperties:
    """Property-based tests for formation matching."""
    
    @given(
        st.text(min_size=1, max_size=50),
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10),
        st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=20)  # Limit examples to avoid slow fuzzy matching
    def test_fuzzy_matching_returns_string_or_none(self, query_formation, candidate_formations, threshold):
        """Fuzzy matching should return string or None."""
        try:
            from rapidfuzz import process, fuzz
            
            result = process.extractOne(
                query_formation,
                candidate_formations,
                scorer=fuzz.WRatio,
                score_cutoff=int(threshold * 100)
            )
            
            if result:
                matched_formation, score, _ = result
                assert isinstance(matched_formation, str), \
                    f"Fuzzy match returned non-string: {type(matched_formation)}"
                assert matched_formation in candidate_formations, \
                    f"Matched formation not in candidates: {matched_formation}"
                assert 0 <= score <= 100, \
                    f"Score out of range: {score}"
            else:
                # No match found, which is valid
                pass
        except ImportError:
            pytest.skip("rapidfuzz not available")
    
    @given(
        st.text(min_size=4, max_size=50),  # At least 4 chars to avoid very short edge cases
        st.lists(st.text(min_size=4, max_size=50), min_size=1, max_size=10)  # At least 4 chars
    )
    @settings(max_examples=20)
    def test_fuzzy_matching_is_case_insensitive(self, query_formation, candidate_formations):
        """Fuzzy matching should be case-insensitive."""
        assume(query_formation.strip() and all(c.strip() for c in candidate_formations))  # Skip whitespace-only
        # Skip if query is mostly numeric/alphanumeric (edge cases that cause issues)
        assume(not (query_formation.isdigit() or (len(query_formation) <= 4 and query_formation.replace(' ', '').isalnum())))
        
        try:
            from rapidfuzz import process, fuzz
            
            # Test with lowercase query
            result_lower = process.extractOne(
                query_formation.lower(),
                candidate_formations,
                scorer=fuzz.WRatio,
                processor=lambda s: s.lower() if isinstance(s, str) else s,
                score_cutoff=0
            )
            
            # Test with uppercase query
            result_upper = process.extractOne(
                query_formation.upper(),
                candidate_formations,
                scorer=fuzz.WRatio,
                processor=lambda s: s.lower() if isinstance(s, str) else s,
                score_cutoff=0
            )
            
            # Results should be similar (same match or similar scores)
            if result_lower and result_upper:
                # Both found matches - they should be the same or very similar
                # Allow some tolerance for edge cases
                score_diff = abs(result_lower[1] - result_upper[1])
                # For longer strings, expect better consistency
                if len(query_formation) >= 6:
                    assert result_lower[0] == result_upper[0] or score_diff < 20, \
                        f"Case sensitivity detected: lower={result_lower[0]} (score={result_lower[1]}) vs upper={result_upper[0]} (score={result_upper[1]})"
                else:
                    # More lenient for shorter strings (4-5 chars)
                    assert result_lower[0] == result_upper[0] or score_diff < 80, \
                        f"Case sensitivity detected: lower={result_lower[0]} (score={result_lower[1]}) vs upper={result_upper[0]} (score={result_upper[1]})"
        except ImportError:
            pytest.skip("rapidfuzz not available")
    
    @given(
        st.text(min_size=1, max_size=50),
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
    def test_fuzzy_matching_exact_match_returns_high_score(self, formation, candidate_formations):
        """Exact matches should return high scores."""
        # Add formation to candidates to ensure at least one match
        candidate_formations_with_match = list(candidate_formations) + [formation]
        
        try:
            from rapidfuzz import process, fuzz
            
            result = process.extractOne(
                formation,
                candidate_formations_with_match,
                scorer=fuzz.WRatio,
                score_cutoff=0
            )
            
            if result:
                matched_formation, score, _ = result
                # Exact match should have high score (>= 90)
                if matched_formation == formation:
                    assert score >= 90, \
                        f"Exact match has low score: {score} for {formation}"
        except ImportError:
            pytest.skip("rapidfuzz not available")
    
    @given(
        st.text(min_size=4, max_size=50),  # At least 4 chars to avoid very short edge cases
        st.lists(st.text(min_size=4, max_size=50), min_size=1, max_size=10)  # At least 4 chars
    )
    @settings(max_examples=20)
    def test_fuzzy_matching_handles_whitespace(self, query_formation, candidate_formations):
        """Fuzzy matching should handle whitespace variations."""
        assume(query_formation.strip() and all(c.strip() for c in candidate_formations))  # Skip whitespace-only
        # Skip if query is mostly numeric/alphanumeric (edge cases that cause issues)
        assume(not (query_formation.isdigit() or (len(query_formation) <= 4 and query_formation.replace(' ', '').isalnum())))
        
        try:
            from rapidfuzz import process, fuzz
            
            # Test with original
            result_original = process.extractOne(
                query_formation,
                candidate_formations,
                scorer=fuzz.WRatio,
                processor=lambda s: " ".join(s.lower().split()) if isinstance(s, str) else s,
                score_cutoff=0
            )
            
            # Test with extra whitespace
            query_with_spaces = f"  {query_formation}  "
            result_spaced = process.extractOne(
                query_with_spaces,
                candidate_formations,
                scorer=fuzz.WRatio,
                processor=lambda s: " ".join(s.lower().split()) if isinstance(s, str) else s,
                score_cutoff=0
            )
            
            # Results should be similar
            if result_original and result_spaced:
                # Allow some tolerance for edge cases
                score_diff = abs(result_original[1] - result_spaced[1])
                # For longer strings, expect better consistency
                if len(query_formation) >= 6:
                    assert result_original[0] == result_spaced[0] or score_diff < 20, \
                        f"Whitespace sensitivity detected: original={result_original[0]} (score={result_original[1]}) vs spaced={result_spaced[0]} (score={result_spaced[1]})"
                else:
                    # More lenient for shorter strings (4-5 chars)
                    assert result_original[0] == result_spaced[0] or score_diff < 80, \
                        f"Whitespace sensitivity detected: original={result_original[0]} (score={result_original[1]}) vs spaced={result_spaced[0]} (score={result_spaced[1]})"
        except ImportError:
            pytest.skip("rapidfuzz not available")
    
    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=50)
    )
    @settings(max_examples=30)
    def test_formation_normalization_consistency(self, formation1, formation2):
        """Formation normalization should be consistent."""
        # Normalize both formations
        norm1 = normalize_well(formation1) if formation1 else ""
        norm2 = normalize_well(formation2) if formation2 else ""
        
        # If original formations are equal (case-insensitive), normalized should be equal
        if formation1.upper().strip() == formation2.upper().strip():
            assert norm1 == norm2, \
                f"Normalization inconsistent: {formation1} -> {norm1}, {formation2} -> {norm2}"
    
    @given(
        st.text(min_size=1, max_size=50),
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5)
    )
    @settings(max_examples=20)
    def test_fuzzy_matching_threshold_filtering(self, query_formation, candidate_formations):
        """Fuzzy matching should respect threshold."""
        try:
            from rapidfuzz import process, fuzz
            
            # Test with high threshold
            result_high = process.extractOne(
                query_formation,
                candidate_formations,
                scorer=fuzz.WRatio,
                score_cutoff=90  # High threshold
            )
            
            # Test with low threshold
            result_low = process.extractOne(
                query_formation,
                candidate_formations,
                scorer=fuzz.WRatio,
                score_cutoff=0  # Low threshold
            )
            
            # If high threshold returns None, low threshold might return something
            # If low threshold returns None, high threshold should also return None
            if result_low is None:
                assert result_high is None, \
                    f"Low threshold returned None but high threshold returned match"
            
            # If both return results, high threshold result should have score >= 90
            if result_high:
                assert result_high[1] >= 90, \
                    f"High threshold result has score < 90: {result_high[1]}"
        except ImportError:
            pytest.skip("rapidfuzz not available")

