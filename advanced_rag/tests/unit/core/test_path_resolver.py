"""
Unit tests for path resolution.
"""
import pytest
from pathlib import Path
from src.core.path_resolver import PathResolver
from src.core.result import Result


@pytest.mark.unit
class TestPathResolver:
    """Test PathResolver class."""
    
    def test_resolve_vectorstore_with_base_path(self, tmp_path):
        """Test resolve_vectorstore() with explicit base path."""
        test_path = tmp_path / "vectorstore"
        test_path.mkdir()
        
        resolved = PathResolver.resolve_vectorstore(test_path)
        assert resolved == test_path.resolve()
    
    def test_resolve_vectorstore_from_config(self, mock_config, tmp_path):
        """Test resolve_vectorstore() uses config."""
        test_path = tmp_path / "vectorstore"
        test_path.mkdir()
        mock_config.persist_directory = test_path
        
        resolved = PathResolver.resolve_vectorstore()
        assert resolved == test_path.resolve()
    
    def test_resolve_cache_path(self, tmp_path):
        """Test resolve_cache_path() creates correct path."""
        base = tmp_path / "vectorstore"
        base.mkdir()
        
        cache_path = PathResolver.resolve_cache_path("test_cache.json", base)
        assert cache_path == (base / "test_cache.json").resolve()
    
    def test_resolve_documents_with_existing_path(self, tmp_path):
        """Test resolve_documents() with existing path."""
        docs_path = tmp_path / "documents"
        docs_path.mkdir()
        
        result = PathResolver.resolve_documents(docs_path)
        assert result.is_ok()
        assert result.unwrap() == docs_path.resolve()
    
    def test_resolve_documents_with_nonexistent_path(self, tmp_path):
        """Test resolve_documents() returns error for nonexistent path."""
        nonexistent = tmp_path / "nonexistent"
        
        result = PathResolver.resolve_documents(nonexistent)
        assert result.is_err()
        assert "does not exist" in result.error().message
    
    def test_resolve_well_picks_dat(self, tmp_path):
        """Test resolve_well_picks_dat() finds .dat file."""
        docs_path = tmp_path / "documents"
        docs_path.mkdir()
        dat_file = docs_path / "Well_picks_Volve_v1.dat"
        dat_file.write_text("test data")
        
        result = PathResolver.resolve_well_picks_dat(docs_path)
        assert result.is_ok()
        assert result.unwrap() == dat_file.resolve()
    
    def test_resolve_well_picks_dat_not_found(self, tmp_path):
        """Test resolve_well_picks_dat() returns error if not found."""
        docs_path = tmp_path / "documents"
        docs_path.mkdir()
        # Don't create .dat file
        
        result = PathResolver.resolve_well_picks_dat(docs_path)
        assert result.is_err()
        assert "not found" in result.error().message.lower()

