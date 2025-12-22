"""
Unit tests for tool adapter module.
"""
import pytest
import json
from src.core.tool_adapter import result_to_string, tool_wrapper
from src.core.result import Result, AppError, ErrorType


@pytest.mark.unit
class TestResultToString:
    """Test result_to_string function."""
    
    def test_returns_value_for_ok_result(self):
        """Test returns value for successful Result."""
        result = Result.ok("success_value")
        output = result_to_string(result)
        assert output == "success_value"
    
    def test_returns_json_error_for_error_result(self):
        """Test returns JSON error for error Result."""
        result = Result.err(AppError(
            type=ErrorType.PROCESSING_ERROR,
            message="Error occurred"
        ))
        output = result_to_string(result)
        
        # Should be JSON string
        error_data = json.loads(output)
        assert "error" in error_data
        assert "message" in error_data
        assert error_data["message"] == "Error occurred"
    
    def test_includes_details_in_error_json(self):
        """Test includes details in error JSON if present."""
        result = Result.err(AppError(
            type=ErrorType.PROCESSING_ERROR,
            message="Error occurred",
            details={"key": "value"}
        ))
        output = result_to_string(result)
        
        error_data = json.loads(output)
        assert "details" in error_data
        assert error_data["details"]["key"] == "value"
    
    def test_uses_default_error_message(self):
        """Test uses default error message when provided."""
        result = Result.err(AppError(
            type=ErrorType.PROCESSING_ERROR,
            message="Custom error"
        ))
        output = result_to_string(result, default_error="Default message")
        
        # Should still use actual error message, not default
        error_data = json.loads(output)
        assert error_data["message"] == "Custom error"


@pytest.mark.unit
class TestToolWrapper:
    """Test tool_wrapper decorator."""
    
    def test_wraps_result_function_to_return_string(self):
        """Test wraps Result-returning function to return string."""
        @tool_wrapper
        def test_tool(query: str) -> Result[str, AppError]:
            return Result.ok(f"Answer: {query}")
        
        result = test_tool("test query")
        assert isinstance(result, str)
        assert result == "Answer: test query"
    
    def test_wraps_error_result_to_json_string(self):
        """Test wraps error Result to JSON string."""
        @tool_wrapper
        def failing_tool(query: str) -> Result[str, AppError]:
            return Result.err(AppError(
                type=ErrorType.NOT_FOUND_ERROR,
                message="Not found"
            ))
        
        result = failing_tool("test")
        assert isinstance(result, str)
        error_data = json.loads(result)
        assert error_data["error"] == ErrorType.NOT_FOUND_ERROR.value
    
    def test_preserves_function_name(self):
        """Test preserves original function name."""
        @tool_wrapper
        def my_tool(query: str) -> Result[str, AppError]:
            return Result.ok("result")
        
        assert my_tool.__name__ == "my_tool"
    
    def test_preserves_function_docstring(self):
        """Test preserves original function docstring."""
        @tool_wrapper
        def documented_tool(query: str) -> Result[str, AppError]:
            """This is a documented tool."""
            return Result.ok("result")
        
        assert "documented tool" in documented_tool.__doc__
    
    def test_handles_function_with_args(self):
        """Test handles function with arguments."""
        @tool_wrapper
        def tool_with_args(x: int, y: int) -> Result[str, AppError]:
            return Result.ok(str(x + y))
        
        result = tool_with_args(2, 3)
        assert result == "5"
    
    def test_handles_function_with_kwargs(self):
        """Test handles function with keyword arguments."""
        @tool_wrapper
        def tool_with_kwargs(x: int, y: int = 10) -> Result[str, AppError]:
            return Result.ok(str(x + y))
        
        result = tool_with_kwargs(5, y=20)
        assert result == "25"

