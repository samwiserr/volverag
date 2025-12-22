"""
Unit tests for decorators module.
"""
import pytest
from src.core.decorators import handle_errors, to_result
from src.core.result import Result, AppError, ErrorType


@pytest.mark.unit
class TestHandleErrors:
    """Test handle_errors decorator."""
    
    def test_handles_successful_function(self):
        """Test decorator returns Result.ok for successful function."""
        @handle_errors(ErrorType.PROCESSING_ERROR)
        def successful_func() -> str:
            return "success"
        
        result = successful_func()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() == "success"
    
    def test_handles_exception(self):
        """Test decorator converts exception to Result.err."""
        @handle_errors(ErrorType.PROCESSING_ERROR, "Custom error message")
        def failing_func():
            raise ValueError("Something went wrong")
        
        result = failing_func()
        assert isinstance(result, Result)
        assert result.is_err()
        error = result.error()
        assert error.type == ErrorType.PROCESSING_ERROR
        assert "Something went wrong" in error.message or "Custom error message" in error.message
    
    def test_preserves_result_return(self):
        """Test decorator preserves Result if function already returns Result."""
        @handle_errors(ErrorType.PROCESSING_ERROR)
        def result_func() -> Result[str, AppError]:
            return Result.ok("already a result")
        
        result = result_func()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() == "already a result"
    
    def test_preserves_error_result(self):
        """Test decorator preserves error Result."""
        @handle_errors(ErrorType.PROCESSING_ERROR)
        def error_func() -> Result[str, AppError]:
            return Result.err(AppError(
                type=ErrorType.NOT_FOUND_ERROR,
                message="Not found"
            ))
        
        result = error_func()
        assert isinstance(result, Result)
        assert result.is_err()
        assert result.error().type == ErrorType.NOT_FOUND_ERROR
    
    def test_includes_function_context(self):
        """Test decorator includes function name in error context."""
        @handle_errors(ErrorType.PROCESSING_ERROR)
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        assert result.is_err()
        error = result.error()
        assert "function" in error.details or "function" in str(error.context)
    
    def test_handles_function_with_args(self):
        """Test decorator works with functions that have arguments."""
        @handle_errors(ErrorType.PROCESSING_ERROR)
        def func_with_args(x: int, y: int) -> int:
            return x + y
        
        result = func_with_args(2, 3)
        assert result.is_ok()
        assert result.unwrap() == 5
    
    def test_handles_function_with_kwargs(self):
        """Test decorator works with keyword arguments."""
        @handle_errors(ErrorType.PROCESSING_ERROR)
        def func_with_kwargs(x: int, y: int = 10) -> int:
            return x + y
        
        result = func_with_kwargs(5, y=20)
        assert result.is_ok()
        assert result.unwrap() == 25


@pytest.mark.unit
class TestToResult:
    """Test to_result decorator."""
    
    def test_converts_return_value_to_result(self):
        """Test decorator converts return value to Result.ok."""
        @to_result(ErrorType.PROCESSING_ERROR)
        def successful_func() -> str:
            return "success"
        
        result = successful_func()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() == "success"
    
    def test_preserves_existing_result(self):
        """Test decorator preserves Result if already returned."""
        @to_result(ErrorType.PROCESSING_ERROR)
        def result_func() -> Result[str, AppError]:
            return Result.ok("already result")
        
        result = result_func()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() == "already result"
    
    def test_converts_exception_to_result(self):
        """Test decorator converts exception to Result.err."""
        @to_result(ErrorType.PROCESSING_ERROR)
        def failing_func():
            raise ValueError("Error occurred")
        
        result = failing_func()
        assert isinstance(result, Result)
        assert result.is_err()
        error = result.error()
        assert error.type == ErrorType.PROCESSING_ERROR
    
    def test_default_error_type(self):
        """Test decorator uses default error type when not specified."""
        @to_result()
        def failing_func():
            raise RuntimeError("Default error")
        
        result = failing_func()
        assert result.is_err()
        error = result.error()
        assert error.type == ErrorType.PROCESSING_ERROR  # Default
    
    def test_handles_none_return(self):
        """Test decorator handles None return value."""
        @to_result(ErrorType.PROCESSING_ERROR)
        def none_func():
            return None
        
        result = none_func()
        assert isinstance(result, Result)
        assert result.is_ok()
        assert result.unwrap() is None

