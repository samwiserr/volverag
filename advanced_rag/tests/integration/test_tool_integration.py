"""
Integration tests for tool orchestration.

Tests that tools work together correctly and handle errors properly.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import json

from src.core.result import Result, AppError, ErrorType
from src.core.well_utils import normalize_well, extract_well


@pytest.mark.integration
class TestToolIntegration:
    """Test tool integration and orchestration."""
    
    def test_well_utils_integration(self):
        """Test well utilities work correctly."""
        # Normalize well names
        assert normalize_well("15/9-F-5") == "159F5"
        # normalize_well removes all non-alphanumeric, so "WELL NO" becomes part of the result
        normalized = normalize_well("WELL NO 15/9-F-15 A")
        assert "159F15A" in normalized  # Should contain the well part
        
        # Extract well from text
        text = "What is porosity in well 15/9-F-5?"
        well = extract_well(text)
        assert well is not None
        assert "15/9-F-5" in well or "159F5" in well
    
    def test_petro_params_tool_structure(self, mock_config, tmp_path):
        """Test petro params tool structure."""
        from src.tools.petro_params_tool import PetroParamsTool
        import json
        
        # Create a temporary cache file
        cache_file = tmp_path / "petro_params_cache.json"
        cache_data = {"rows": []}
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        
        # Tool should be initializable with cache_path
        tool = PetroParamsTool(cache_path=str(cache_file))
        assert tool is not None
        assert hasattr(tool, "get_tool")
    
    def test_validation_integration(self, mock_config):
        """Test validation works across the system."""
        from src.core.validation import validate_query
        from src.core.security import sanitize_input
        
        # Test query validation
        is_valid, error = validate_query("test")
        assert is_valid
        
        # Test sanitization
        result = sanitize_input("normal query")
        assert result.is_ok()
        
        # Test dangerous input
        result = sanitize_input("<script>alert('xss')</script>")
        assert result.is_err()
    
    def test_result_pattern_integration(self):
        """Test Result pattern works across tools."""
        from src.core.result import Result, AppError, ErrorType
        
        # Success case
        result = Result.ok("success")
        assert result.is_ok()
        assert result.unwrap() == "success"
        
        # Error case
        error = AppError(ErrorType.VALIDATION_ERROR, "Invalid")
        result = Result.err(error)
        assert result.is_err()
        assert result.error().type == ErrorType.VALIDATION_ERROR
        
        # Chaining
        result = Result.ok(10)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok()
        assert mapped.unwrap() == 20
    
    def test_config_integration(self, monkeypatch):
        """Test configuration works across modules."""
        from src.core.config import get_config, reload_config
        
        # Set test env vars
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        
        reload_config()
        config = get_config()
        
        assert config.openai_api_key == "test-key"
        assert config.llm_model == "gpt-4o"

